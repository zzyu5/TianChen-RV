#ifndef TIANCHENRV_CONVERSION_RVV_RVVTOEMITCINTERNAL_H
#define TIANCHENRV_CONVERSION_RVV_RVVTOEMITCINTERNAL_H

// Implementation-internal header for the RVV->EmitC variant lowering. This is
// NOT a public API: it exists solely so the VariantToEmitCFunc conversion
// pattern's ~166 method definitions can be split across several family-grouped
// translation units (block-quant-linear, K-quant, codebook/fp4, grid-codebook,
// ternary/binary, forward-elementwise, deferred/dequant/masked-store) that
// compile in parallel. To define a class's methods out-of-line in a different
// TU the class must have a NAMED-namespace identity (an anonymous-namespace
// class yields a distinct type per TU), so VariantToEmitCFunc lives in
// `tianchenrv::conversion::rvv::detail` here -- mirroring the support helpers
// already extracted to RVVToEmitCSupport.{h,cpp}. The emitted C is byte-
// identical to the former monolithic translation unit (a pure code move): the
// block-dot conversion fingerprint and the lit suite prove it.

#include "TianChenRV/Conversion/RVV/RVVToEmitCSupport.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace detail {

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;
namespace emitc = ::mlir::emitc;

/// The scalar fp16->fp32 read callee spelling used by the typed
/// tcrv_rvv.block_fp16_scale_product lowering. It currently holds the SAME
/// string as the monolithic block-dot emitters' independent local
/// `fp16ReadCallee` literals (RVVToEmitCBlockQuantLinear.cpp:187, :479, :5428,
/// :6022), so the emitted C is byte-EQUAL to the monolith's inline fp16 read.
/// That equality is a byte-equal-literal coincidence, NOT a mechanized
/// single-source share: those monolith literals do NOT reference this constant
/// (only the brick lowering does). Consolidating them onto this constant --
/// which is what would actually mechanize drift-protection -- is deferred to
/// brick(5) (the q8_0 wire-in), where the monolith inline reads are replaced by
/// the typed op.
inline constexpr llvm::StringRef kFp16ScaleReadCallee =
    "(float)*(const _Float16 *)";

struct AbiParam {
  tcrvrvv::RuntimeABIValueOp op;
  std::string cType;
  mlir::Type emitcType;
};

//===----------------------------------------------------------------------===//
// Flat-plain block-dot body descriptor (Fork B: descriptor-driven emitter body).
//===----------------------------------------------------------------------===//

// The per-block INTEGER-CORE decode primitive selected by descriptor field
// rather than op identity. Each value names the already-factored decode helper
// the shared skeleton dispatches to inside emitStripReduce:
//   PlainI8            -> inline vwmul i8xi8 (q8_0, no decode)
//   OffsetBinaryNibble -> emitOffsetBinaryDecodeProductValue (q4_0, xor 0x88 +
//                         sll/sra sign-extend + vwmul/vwmacc; weight loads i8)
//   UnsignedNibble     -> emitUnsignedNibbleDecodeProductValue (q4_1, vand 0x0F
//                         + vsrl 0x04 + reinterpret; weight loads u8)
//   FiveBitOffsetBinary-> emitFiveBitOffsetBinaryDecodeProductValue (q5_0/q5_1,
//                         nibble + qh 5th-bit merge; weight loads u8; the qh
//                         field + activation_quant_byte_offset are read)
//   CodebookGatherNibble-> the 2nd primitive class (Fork B codebook extension):
//                         the two UNSIGNED nibble index lanes (vand 0x0F / vsrl
//                         0x04) GATHER through a broadcast 16-entry int8 codebook
//                         table (vrgather_vv_i8<L>(values, idx)) into signed-i8
//                         weight lanes, then feed the SAME
//                         emitOffsetBinaryProductFromDecodedValue product chain
//                         the offset-binary sibling uses (iq4_nl / mxfp4). The
//                         table register `values` is broadcast-loaded ONCE above
//                         the block loop from the descriptor's codebook entries.
enum class FlatDecodePrimitive {
  PlainI8,
  OffsetBinaryNibble,
  UnsignedNibble,
  FiveBitOffsetBinary,
  CodebookGatherNibble,
};

// The per-block WEIGHT scale source. Fp16 is the sanctioned `(float)*(const
// _Float16 *)` read every flat-plain + iq4_nl format uses; E8M0 is the mxfp4
// FP4-class structured reconstruction `GGML_E8M0_TO_FP32_HALF(e) = 2^(e-128)`
// from a single shared-exponent byte (no fp16 field). The activation scale is
// always the fp16 read. Selected by descriptor field, not op identity.
enum class FlatWeightScaleSource {
  Fp16,
  E8M0,
};

// The per-block fp32 fold model. The fold is grouped into ONE emitc.expression
// so mlir-translate renders a single C statement whose fp associativity /
// contraction matches ggml byte-for-byte, so the fold tree MUST be explicit
// (fp non-associativity). There are FOUR distinct trees in-tree -- q5_0's
// ScalesTimesSumi is a DISTINCT emitc sequence from q8_0's SumiTimesScales even
// though both carry scale_model "dual-fp16-per-block-d_x.d_y" (the operand +
// emission order of the outer mul differ), so fold_model keys off `kind`, not
// `scale_model`:
//   SumiTimesScales -> sumf + (float)sumi * (d_x * d_y)   (q8_0 MONOLITH)
//   LeftAssoc       -> sumf + (float)sumi * d_x * d_y      (q4_0)
//   ScalesTimesSumi -> sumf + (d_x * d_y) * (float)sumi    (q5_0)
//   ScalePlusMin    -> sumf + ((d_x*d_y)*sumi + m_x*s_y)   (q4_1 / q5_1)
//   SeparatedLeftAssoc -> t = (float)sumi * d_x; t = t * d_y; sumf = sumf + t;
//     (the pinned flat-block-dot fp-fold oracle [K-5]/[testing/
//     flat-block-dot-fp-fold-oracle.md §1]: strict left-assoc, ordered, NO
//     dx*dy premultiply, NO FMA contraction. Emitted as SEPARATE emitc
//     statements -- standalone mul/mul/add, NOT one emitc.expression -- so
//     clang's default -ffp-contract=on cannot fuse (t*dy)+sumf into fmaf.
//     Currently the q8_0 TYPED loop body; q4_0/q5_0 conform in step 4.)
enum class FlatFoldModel {
  SumiTimesScales,
  LeftAssoc,
  ScalesTimesSumi,
  ScalePlusMin,
  SeparatedLeftAssoc,
};

// The block-format + primitive facts the shared emitFlatBlockDot body reads to
// generate a flat-plain (q4_0/q8_0/q4_1/q5_0/q5_1) block-dot. The Group-A
// geometry fields mirror the typed op attrs (I4); the Group-B primitive/fold
// fields lift the decode-primitive + fold-model selection from op-identity to
// descriptor fields. deriveFlatBlockDotDescriptor builds it from the op's
// `kind` + attr-presence; the LMUL / unroll / elision SCHEDULE facts stay in
// BlockDotFacts (deriveBlockDotFacts), read separately by the caller.
struct FlatBlockDotDescriptor {
  FlatDecodePrimitive decodePrimitive = FlatDecodePrimitive::PlainI8;
  FlatFoldModel foldModel = FlatFoldModel::SumiTimesScales;
  // The i8 integer-core LMUL anchor floor for an attr-less op ("m2" for q8_0's
  // whole 32-element block, "m1" for the nibble half-blocks). Passed to
  // deriveBlockDotFacts by the caller.
  llvm::StringRef defaultCoreLmul = "m2";
  int64_t qk = 0;
  int64_t weightStride = 0;
  int64_t activationStride = 0;
  // Weight quant byte offset (past the fp16 scale). The activation quant offset
  // is a SEPARATE attr for q5_0/q5_1 and otherwise reuses quantOffset.
  int64_t quantOffset = 0;
  int64_t activationQuantOffset = 0;
  // qk (whole-block, plain_i8) or qk/2 (nibble half-block) -- the strip length.
  int64_t blockLen = 0;
  // The q8 high-half byte offset (used by the two-activation-load primitives).
  int64_t highOffset = 0;
  // The per-element 5th-bit qh field (five_bit primitive only).
  bool hasQh = false;
  int64_t qhOffset = 0;
  // The 5-bit offset-binary `-16` bias: true for q5_0, false for q5_1 (its bias
  // lives in the per-block MIN scale, like q4_1).
  bool applyOffsetBias = true;
  // The Family-B MIN/SUM correction: read m_x/s_y and fold the second product.
  bool hasMinTerm = false;
  int64_t weightMinOffset = 0;
  int64_t activationSumOffset = 0;
  // The 2nd primitive class (codebook) extension. When hasCodebook is set the
  // shared body emits a `static const int8_t <tableName>[N]` decl + a `vle8`
  // broadcast of `codebook` into a table register ONCE above the block loop, and
  // the CodebookGatherNibble decode gathers through it. weightScaleSource selects
  // the fp16 (iq4_nl) vs E8M0 (mxfp4) weight-scale read.
  bool hasCodebook = false;
  llvm::ArrayRef<int8_t> codebook;
  llvm::StringRef codebookTableName;
  FlatWeightScaleSource weightScaleSource = FlatWeightScaleSource::Fp16;
};

// Build a FlatBlockDotDescriptor from a GgmlBlockDot* op's `kind` string + its
// block-format attrs (attr-presence for the optional-by-format ones). This is
// the emitter-side mirror of the front-door family table + deriveBlockDotFacts:
// the `kind` selects the decode primitive / fold model / core-LMUL floor, and
// the I4 geometry attrs fill the block-format fields. Returns std::nullopt for a
// non-flat-plain kind (the caller keeps its bespoke emitter).
std::optional<FlatBlockDotDescriptor>
deriveFlatBlockDotDescriptor(mlir::Operation *op);

// The shared per-block emit state the factored flat-block-dot core methods
// (emitFlatBlockCore / emitFlatIntegerCore / emitFlatFold) read. It bundles the
// block-format descriptor + the scheduled facts (coreLmul / wideLmul / elision)
// + the interned emitc types + the ABI base pointers + the sumf accumulator
// lvalue, so the per-block integer-core arithmetic + fp32-fold guts live as ONE
// SHARED typed lowering that BOTH the monolithic emitFlatBlockDot AND the M-FLAT
// tcrv_rvv.typed_flat_block_dot_loop_body region driver invoke. It is NOT
// entangled with any GgmlBlockDot* op entry: a step-6 monolith delete removes
// the GgmlBlockDot* op + dispatch + orchestration and these guts survive as the
// typed body's lowering (the honest-LOC boundary).
struct FlatBlockDotEmitState {
  FlatBlockDotDescriptor descriptor;
  llvm::StringRef opName;
  llvm::StringRef role;
  llvm::StringRef coreLmul;
  llvm::StringRef wideLmul;
  bool stripElided = false;
  mlir::Value weightBase;
  mlir::Value activationBase;
  mlir::Value sumfVar;
  mlir::Value codebookValues; // null unless the descriptor carries a codebook
  mlir::Type sizeType;
  mlir::Type floatType;
  mlir::Type i32Type;
  mlir::Type u32Type;
  mlir::Type weightPtrType;
  mlir::Type activationPtrType;
  mlir::Type i8CoreType;
  mlir::Type u8CoreType;
  mlir::Type i16WideType;
  mlir::Type u16WideType;
  mlir::Type i32m1Type;
  mlir::Type i8PtrType;
  mlir::Type u8PtrType;
  llvm::StringRef fp16ReadCallee;
  llvm::StringRef u16ReadCallee;
};

// The per-block integer core + fp16 scale reads emitFlatBlockCore returns; the
// caller folds them in strict ascending block order (fp non-associativity).
struct FlatBlockCore {
  mlir::Value sumiVar;
  mlir::Value dX;
  mlir::Value dY;
  mlir::Value mX;
  mlir::Value sY;
};

class VariantToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::exec::VariantOp> {
public:
  VariantToEmitCFunc(const mlir::TypeConverter &typeConverter,
                     mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tcrv::exec::VariantOp>(typeConverter,
                                                         context) {}

  mlir::LogicalResult
  matchAndRewrite(tcrv::exec::VariantOp variant, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override;

private:
  /// Emit one selected-body VL scope as a runtime setvl-tail chunk loop:
  ///   for (size_t i = 0; i < n; i += vlmax) { vl = setvl(n - i); <body> }
  /// walking the scope's typed dataflow ops in (reorder-adjusted) IR order.
  /// This is the per-scope emission shared by single-scope families (called
  /// once) and the multi-scope Gearbox dequant body (called per scope). It does
  /// NOT emit the function epilogue/return; the caller owns that.
  mlir::LogicalResult emitScopeForLoop(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
          &segmentFieldMap,
      bool standaloneReduction) const;

  /// True iff `scope` is a low-precision Gearbox product-reduce-dequantize body:
  /// it dequantizes an i32 accumulator built from a signed widening or packed-i4
  /// nibble-unpack product reduced over i8mf4 sources, then stores the f32
  /// result. Structurally: the scope carries at least one tcrv_rvv.dequantize
  /// whose source is a tcrv_rvv.standalone_reduce of a tcrv_rvv.widening_product
  /// or tcrv_rvv.packed_i4_nibble_unpack_product, plus a store. This is the
  /// dequant family the dedicated multi-loop + function-scoped accumulator
  /// routine owns; every other body falls to the single-scope emitScopeForLoop.
  static bool isLowPrecisionDequantBody(tcrvrvv::WithVLOp scope);

  /// True iff `scope` is the DEFERRED-WIDE low-precision contraction body (the
  /// N3 resource-aware max-legal-LMUL schedule, the measured ssh-rvv winner
  /// var_v_m2_a1.c): the dequant sources a trailing tcrv_rvv.standalone_reduce
  /// whose input is a tcrv_rvv.widening_accumulate (the i32m8 deferred vector
  /// accumulate). The widening_accumulate op is the STRUCTURAL marker (I5): the
  /// conversion emits the deferred-wide algorithm because the op is in the body,
  /// not because of any metadata. A body with the narrow per-iteration
  /// vwredsum-into-scalar reduce (isLowPrecisionDequantBody) has NO
  /// widening_accumulate, so the two recognizers are disjoint.
  static bool isDeferredWideDequantBody(tcrvrvv::WithVLOp scope);

  /// True iff `scope` is the 2nd-family (i16 dot-reduce) DEFERRED-WIDE body: a
  /// tcrv_rvv.store whose stored value is a tcrv_rvv.standalone_reduce (i32m8 ->
  /// i32m1) whose input is a tcrv_rvv.deferred_accumulate (the i32m8 NON-widening
  /// vadd.vv deferred accumulate). The deferred_accumulate op is the STRUCTURAL
  /// marker (I5): emission follows op identity. Disjoint from the byte path
  /// (which has a WideningAccumulateOp + a DequantizeOp) and from the narrow
  /// per-iter dot-reduce (which has a WideningDotReduceOp, no deferred_accumulate).
  /// True iff `scope` is the ggml Q4_0 x Q8_0 block dot-product body: a with_vl
  /// scope whose ONLY compute op is a single tcrv_rvv.q4_0_q8_0_block_dot (the
  /// whole AoS block loop + per-block dual fp16 scale + fp32 accumulation + *s
  /// store is the op's emission, not a body of sub-ops). The op identity is the
  /// dispatch key; the emitter owns the structured expansion.
  static bool isQ4_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The GEMM-tile (weight-decode reuse) recognizer: a with_vl scope whose ONLY
  /// compute op is a single tcrv_rvv.q4_0_q8_0_gemm_tile. The op identity is the
  /// dispatch key; the emitter owns the structured weight-reuse expansion.
  static bool isQ4_0Q8_0GemmTileBody(tcrvrvv::WithVLOp scope);

  /// The full-GEMM (NR rows x nc cols) recognizer: a with_vl scope whose ONLY
  /// compute op is a single tcrv_rvv.q4_0_q8_0_gemm. The op identity is the
  /// dispatch key; the emitter owns the structured row x column-strip expansion.
  static bool isQ4_0Q8_0GemmBody(tcrvrvv::WithVLOp scope);

  /// The q5_0 16x1-REPACKED single-column GEMV (decode) recognizer: a with_vl
  /// scope whose ONLY compute op is a single tcrv_rvv.repack_gemv_q5_0_q8_0. The
  /// op identity is the dispatch key; the emitter owns the block-as-lane
  /// expansion with the transposed bit-packed qh 5th-bit decode.
  static bool isRepackGemvQ5_0Q8_0Body(tcrvrvv::WithVLOp scope);

  /// The option-2 stage-C1b PACK (materialize) recognizer: a with_vl scope whose
  /// ONLY compute op is a single tcrv_rvv.pack_q4_0_to_q4_0x16. The op identity
  /// is the dispatch key; the emitter owns the scalar gather + ^0x88 pack body.
  static bool isPackQ4_0ToX16Body(tcrvrvv::WithVLOp scope);

  /// The FAMILY-B 16x1-REPACKED single-column GEMV (decode) recognizer: a
  /// with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.repack_gemv_q4_1_q8_1. The op identity is the dispatch key; the
  /// emitter owns the structured block-as-lane single-output-column expansion
  /// with the q4_1 scale+MIN fold.
  static bool isRepackGemvQ4_1Q8_1Body(tcrvrvv::WithVLOp scope);

  /// The FAMILY-B 16x1-REPACKED full-GEMM (prefill) recognizer: a with_vl scope
  /// whose ONLY compute op is a single tcrv_rvv.repack_gemm_q4_1_q8_1. The op
  /// identity is the dispatch key; the emitter owns the structured block-as-lane
  /// multi-output-column expansion with the q4_1 scale+MIN per-column fold.
  static bool isRepackGemmQ4_1Q8_1Body(tcrvrvv::WithVLOp scope);
  /// The K-quant (super-block) 16x1-REPACKED multi-output-column GEMM (prefill)
  /// recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.repack_gemm_q4_K_q8_K. The op identity is the dispatch key; the
  /// emitter owns the structured block-as-lane multi-output-column expansion
  /// with the q4_K dual d/dmin 8-sub-block 6-bit scale fold.
  static bool isRepackGemmQ4KQ8KBody(tcrvrvv::WithVLOp scope);
  /// The Family-A (symmetric, full-int8) 16x1-REPACKED single-column GEMV
  /// (decode) recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.repack_gemv_q8_0_q8_0. The op identity is the dispatch key; the
  /// emitter owns the structured block-as-lane single-output-column expansion
  /// with FULL int8 weight lanes (NO nibble decode) and i32 in-block accumulation.
  static bool isRepackGemvQ8_0Q8_0Body(tcrvrvv::WithVLOp scope);

  /// The K-QUANT (super-block) 16x1-REPACKED single-column GEVM recognizer: a
  /// with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.repack_gemv_q4_K_q8_K. The op identity is the dispatch key; the
  /// emitter owns the structured block-as-lane single-output-column expansion
  /// with the q4_K 8-sub-block dual (scale + bsums-min) 6-bit fold.
  static bool isRepackGemvQ4KQ8KBody(tcrvrvv::WithVLOp scope);

  /// The M-FLAT loop-scaffold recognizer: a with_vl scope whose ONLY op is a
  /// single tcrv_rvv.typed_flat_block_dot_loop_body (the region-carrying nb
  /// block loop with a SSA loop-carried f32 accumulator). Routed through the
  /// block-dot table (NOT the elementwise path) so no outer AVL loop wraps it;
  /// the op owns its own internal block loop, exactly like emitFlatBlockDot.
  static bool isTypedFlatBlockDotLoopBody(tcrvrvv::WithVLOp scope);

  /// The M-FLAT q4_K/q5_K super-block loop-scaffold recognizer (milestone-2): a
  /// with_vl scope whose ONLY op is a single
  /// tcrv_rvv.typed_super_block_block_dot_loop_body (the region-carrying nb
  /// super-block loop with the DUAL sums-vector + sumf-scalar loop-carried
  /// accumulator). Routed through the block-dot table (NOT the elementwise path)
  /// so no outer AVL loop wraps it; the op owns its own internal super-block
  /// loop, exactly like the monolithic emitQ4_KQ8_KBlockDot.
  static bool isTypedSuperBlockBlockDotLoopBody(tcrvrvv::WithVLOp scope);

  /// The M-FLAT q4_0 16x1-REPACKED GEVM loop-scaffold recognizer (milestone-1): a
  /// with_vl scope whose ONLY op is a single
  /// tcrv_rvv.typed_repack_gemv_loop_body (the region-carrying nb contraction-
  /// block loop with the per-strip LANE-WISE f32 VECTOR loop-carried accumulator,
  /// wrapped by the emitter in the outer weight-column-group loop). Routed through
  /// the block-dot table (NOT the elementwise path) so no outer AVL loop wraps it;
  /// the op owns its own internal loop nest, exactly like the monolithic
  /// emitRepackGemvQ4_0Q8_0.
  static bool isTypedRepackGemvLoopBody(tcrvrvv::WithVLOp scope);

  /// The M-FLAT q4_0 16x1-REPACKED GEMM loop-scaffold recognizer: a with_vl scope
  /// whose ONLY op is a single tcrv_rvv.typed_repack_gemm_loop_body (the
  /// region-carrying nb contraction-block loop with the per-column per-strip
  /// LANE-WISE f32 VECTOR loop-carried accumulators, wrapped by the emitter in the
  /// outer row-group / column-group / runtime-strip / column-pass loops). Routed
  /// through the block-dot table (NOT the elementwise path) so no outer AVL loop
  /// wraps it; the op owns its own internal loop nest, exactly like the monolithic
  /// emitRepackGemmQ4_0Q8_0.
  static bool isTypedRepackGemmLoopBody(tcrvrvv::WithVLOp scope);

  /// The BINARY-class sibling recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q1_0_q8_0_block_dot. The op identity is the dispatch
  /// key; the emitter owns the structured binary-sign-decode expansion.
  static bool isQ1_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The CODEBOOK SUPER-BLOCK recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.iq4_xs_q8_k_block_dot (the super-block variant of
  /// iq4_nl: iq4_nl's codebook gather + the q4_K-style super-block signed scale).
  static bool isIQ4XSQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  // NOTE: the monolith recognizer isIQ2XXSQ8KBlockDotBody was RETIRED at the iq2_xxs
  // flip (L3 coverage): the front door now constructs the typed super-block
  // SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 66),
  // lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs, which reuses the SHARED
  // byte-exact anchors emitIQ2XXSCanonicalGridTableDecl +
  // emitIQ2XXSCanonicalSigns64TableDecl + emitIQ2XXSSuperBlockGridBody.

  // NOTE: the monolith recognizer isIQ2XSQ8KBlockDotBody was RETIRED at the iq2_xs flip
  // (L3 coverage): the front door now constructs the typed super-block SCALAR-accumulator
  // GRID loop body (fold_model "scalar_delta_grid", stride 74), lowered by
  // emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xs, which reuses the SHARED byte-exact
  // anchors emitIQ2XSCanonicalGridTableDecl + emitIQ2XSCanonicalSigns64TableDecl +
  // emitIQ2XSSuperBlockGridBody.

  // NOTE: the monolith recognizer isIQ2SQ8KBlockDotBody was RETIRED at the iq2_s flip
  // (L3 coverage): the front door now constructs the typed super-block SCALAR-accumulator
  // GRID loop body (fold_model "scalar_delta_grid", stride 82), lowered by
  // emitTypedSuperBlockScalarDeltaGridLoopBodyIq2s, which reuses the SHARED byte-exact
  // anchors emitIQ2SCanonicalGridTableDecl + emitIQ2SCanonicalSigns256TableDecl +
  // emitIQ2SSuperBlockGridBody.

  // NOTE: the monolith recognizer isIQ3XXSQ8KBlockDotBody was RETIRED at the iq3_xxs
  // flip (L3 coverage): the front door now constructs the typed super-block
  // SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 98),
  // lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs (dispatched from
  // emitTypedSuperBlockScalarDeltaGridLoopBody on the iq3_xxs grid-core brick identity).

  /// The GRID-codebook sibling recognizer: a with_vl scope whose ONLY compute op is
  /// a single tcrv_rvv.iq3_s_q8_k_block_dot (a member of the deep IQ tail; a
  /// re-composition of the 512-entry grid-of-4 (iq3_xxs), the qh 9th-bit plane
  /// (iq2_s), the EXPLICIT signs read from memory (iq2_s), and the explicit
  /// per-sub-block 4-bit scales (iq2_s)).
  static bool isIQ3SQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  // NOTE: the monolith IQ1_M recognizer isIQ1MQ8KBlockDotBody was RETIRED at the
  // iq1_m flip (L3): the front door now constructs the typed super-block
  // SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 56),
  // recognized by the loop op + resolved via emitTypedSuperBlockScalarDeltaGridLoopBody.

  /// The FP4 CODEBOOK sibling recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.mxfp4_q8_0_block_dot.
  static bool isMXFP4Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The NVFP4 (NVIDIA FP4) CODEBOOK sibling recognizer: a with_vl scope whose
  /// ONLY compute op is a single tcrv_rvv.nvfp4_q8_0_block_dot.
  static bool isNVFP4Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The K-quant K1 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q6_k_q8_k_aux32_partial (the Q6_K x Q8_K super-block
  /// integer aux32 partial).
  static bool isQ6_KQ8_KAux32PartialBody(tcrvrvv::WithVLOp scope);

  /// The Track B q4_K brick-1 recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q4_k_nibble_unpack (one super-block's Region-A plain
  /// 4-bit nibble unpack into aux8[256], NO bit-dance / dot / fold).
  static bool isQ4_KNibbleUnpackBody(tcrvrvv::WithVLOp scope);

  /// The Track B q4_K brick-2 recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q4_k_scale_min_bit_dance (one super-block's Region-B
  /// 6-bit scale/min bit-dance into utmp[4], NO nibble unpack / dot / fold).
  static bool isQ4_KScaleMinBitDanceBody(tcrvrvv::WithVLOp scope);

  /// The Track B q4_K brick-3 recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q4_k_scaled_dot (one super-block's Region-C
  /// per-sub-block uint6-scaled i32 dot into aux32 + the integer fold-back, NO
  /// nibble unpack / bit-dance / MIN term / fp32 fold).
  static bool isQ4_KScaledDotBody(tcrvrvv::WithVLOp scope);

  /// The Track B q4_K brick-4 recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q4_k_min_term (one super-block's MIN term
  /// `sumf -= dmin * sum(mins * bsums)`, NO nibble unpack / bit-dance / scaled
  /// dot / fp32 positive fold).
  static bool isQ4_KMinTermBody(tcrvrvv::WithVLOp scope);

  /// The Track B q4_K brick-6 recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q4_k_sums_fold_scale_d (one super-block's DEFERRED fp32
  /// POSITIVE fold `sums += fp16(x.d) * y.d * (float)aux32`, NO nibble unpack /
  /// bit-dance / scaled dot / MIN term).
  static bool isQ4_KSumsFoldScaleDBody(tcrvrvv::WithVLOp scope);

  /// The Track B q4_K brick-7 recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q4_k_horizontal_fold (the post-loop horizontal fold
  /// `for (l=0..7) sumf += sums8[l]`, NO nibble unpack / bit-dance / scaled dot /
  /// MIN term / positive fold).
  static bool isQ4_KHorizontalFoldBody(tcrvrvv::WithVLOp scope);

  /// The q4_K K4a recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q4_k_q8_k_aux_partial (the Q4_K x Q8_K super-block integer aux32 +
  /// decoded scale/min partial -- the INTEGER CORE before the fp32 d/dmin fold).
  static bool isQ4_KQ8_KAux32PartialBody(tcrvrvv::WithVLOp scope);

  /// The tq2_0 recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.tq2_0_q8_k_block_dot (the TQ2_0 x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- q2_K's 2-bit weight unpack with the
  /// per-element `-1` ternary bias + a single per-super-block integer
  /// accumulator + the single-fp16-scale SCALAR fp32 fold; NO scales, NO min).
  static bool isTQ2_0Q8_KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The tq1_0 recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.tq1_0_q8_k_block_dot (the TQ1_0 x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- the base-3 trit unpack of the qs[48]
  /// and qh[4] weight arrays into an element-ordered aux8[256], then a single
  /// per-super-block integer accumulator + the single-fp16-scale SCALAR fp32
  /// fold; NO scales, NO min).
  static bool isTQ1_0Q8_KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The forward-pass F1 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.ggml_vec_scale_f32 (the f32 in-place elementwise scale
  /// y[i] *= v). The op identity is the dispatch key; the emitter owns the
  /// structured f32 strip-loop expansion (vle32 / vfmul_vf / vse32).
  static bool isGgmlVecScaleF32Body(tcrvrvv::WithVLOp scope);

  /// The forward-pass F3 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.ggml_rms_norm_f32 (the f32 row rms_norm: Sx^2 scalar-double
  /// reduce -> scalar 1/sqrtf(mean+eps) -> vectorized y[i] = x[i]*scale). The op
  /// identity is the dispatch key; the emitter owns the structured scalar-double
  /// reduction + scalar rsqrt + f32 normalize strip-loop expansion.
  static bool isGgmlRmsNormF32Body(tcrvrvv::WithVLOp scope);

  /// True iff the with_vl body is EXACTLY a single tcrv_rvv.ggml_vec_silu_f32
  /// (the f32 silu: y[i] = x[i]*sigmoid(x[i]), via ggml's EXACT vectorized exp
  /// polynomial). The op identity is the dispatch key; the emitter owns the
  /// structured strip-loop + the node-for-node ggml_v_expf_m2 intrinsic chain.
  static bool isGgmlVecSiluF32Body(tcrvrvv::WithVLOp scope);

  /// True iff the with_vl body is EXACTLY a single tcrv_rvv.ggml_vec_soft_max_f32
  /// (the F5b f32 soft_max: y[i] = e^{x[i]-max}, returning the f64 sum via the
  /// widening reduce). The op identity is the dispatch key; the emitter owns the
  /// structured m2 strip loop with the shared exp polynomial + the loop-carried
  /// f64m1 vfwredusum accumulator + the f64 scalar return.
  static bool isGgmlVecSoftMaxF32Body(tcrvrvv::WithVLOp scope);

  /// True iff the with_vl body is EXACTLY a single tcrv_rvv.quantize_row_q8_0
  /// (the F4 f32->block_q8_0 activation quantizer: per-32-block amax reduction +
  /// scale + f32->i16->i8 narrowing convert + the fp16 d / int8 qs AoS store).
  /// The op identity is the dispatch key; the emitter owns the structured block
  /// loop with the vfredmax reduction, the d?1/d:0 conditional, and the
  /// vfncvt/vncvt narrowing chain.
  static bool isGgmlQuantizeRowQ80Body(tcrvrvv::WithVLOp scope);

  /// The forward-pass F6 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.ggml_rope_norm_f32 (the f32 NORMAL rope: the iterative f32
  /// angle recurrence + scalar libm cosf/sinf cache + the per-pair f32 rotation).
  /// The op identity is the dispatch key; the emitter owns the structured scalar
  /// per-pair loop expansion.
  static bool isGgmlRopeNormF32Body(tcrvrvv::WithVLOp scope);

  static bool isDeferredWideDotReduceBody(tcrvrvv::WithVLOp scope);

  /// Emit one product/reduce slice into the function-scoped accumulator variable
  /// `accVar`: load lhs/rhs at the given pointer offset, widening (or packed-i4
  /// nibble-unpack) product, then reduce into the accumulator (reading the
  /// variable as the running seed and reassigning it). `lhsBuffer`/`rhsBuffer`
  /// are the ABI pointer params; `loadVL` is the slice's runtime VL; `offset`
  /// (may be null) is added to the induction var for the second unroll slice.
  mlir::LogicalResult emitDequantProductReduceSlice(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::LoadOp lhsLoad, tcrvrvv::LoadOp rhsLoad,
      mlir::Operation *productOp, tcrvrvv::StandaloneReduceOp reduce,
      mlir::Value lhsBuffer, mlir::Value rhsBuffer, mlir::Value sliceOffset,
      mlir::Value accVar, tcrvrvv::VectorType accVecType, mlir::Value loadVL,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// One product/reduce slice in the typed dequant body, grouped for emission:
  /// the i8mf4 lhs/rhs loads, the (widening or packed-i4) product op, and the
  /// standalone reduce that folds the product into the accumulator.
  struct DequantSlice {
    tcrvrvv::LoadOp lhsLoad;
    tcrvrvv::LoadOp rhsLoad;
    mlir::Operation *productOp = nullptr;
    tcrvrvv::StandaloneReduceOp reduce;
  };

  /// Emit the low-precision Gearbox product-reduce-dequantize body as a real
  /// emitc function body: a function-scoped i32 accumulator variable seeded from
  /// acc[0], a main chunk loop carrying `unroll_factor` product/reduce slices
  /// (step = vlmax * unroll_factor), a scalar tail loop over the remainder when
  /// unrolled, then the dequant epilogue (scalar extract -> f32 -> *scale ->
  /// store; with an optional clamp via splat/compare/select). The caller has
  /// already emitted the function entry, the runtime-ABI valueMap, the scope
  /// provenance comment, and the pre-loop full-chunk setvl (`vlmax`).
  mlir::LogicalResult emitLowPrecisionDequantBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the DEFERRED-WIDE low-precision contraction body (the N3 resource-aware
  /// max-legal-LMUL schedule, the measured ssh-rvv winner var_v_m2_a1.c) as a
  /// real emitc function body:
  ///   vint32m8_t dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vsetvlmax_e32m8());
  ///   for (i = 0; i < n; i += vlmax) {
  ///     vl = setvl(n - i);
  ///     a = vle8_v_i8m2(lhs + i, vl);  b = vle8_v_i8m2(rhs + i, vl);
  ///     p = vwmul_vv_i16m4(a, b, vl);
  ///     dot_acc_vec = vwadd_wv_i32m8(dot_acc_vec, p, vl);   // DEFERRED accumulate
  ///   }
  ///   vint32m1_t vred = vredsum_vs_i32m8_i32m1(dot_acc_vec, zero, vlmax_acc);
  ///   int32_t sum = acc[0] + vmv_x_s_i32m1_i32(vred);       // SCALAR acc[0] add
  ///   out[0] = (float) sum * scale;                          // dequant epilogue
  /// The i32m8 accumulator is zero-seeded (NOT splatted from acc[0]); acc[0] is
  /// added as a SCALAR after the single trailing vredsum. There is NO
  /// per-iteration vwredsum. The body is body-determined: the structural marker
  /// is tcrv_rvv.widening_accumulate (I5).
  mlir::LogicalResult emitDeferredWideDequantBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The deferred-wide trailing reduction + scalar acc[0] add + dequant store:
  ///   vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vsetvlmax_e32m1());
  ///   vint32m1_t vred  = __riscv_vredsum_vs_i32m8_i32m1(dot_acc_vec, vzero, vlmax_acc);
  ///   int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  ///   float scaled = (float) sum * scale;
  ///   out[0] = vse32(vfmv_v_f(scaled, 1));
  /// The acc[0] seed is added as a SCALAR (the i32m8 accumulator was zero-seeded).
  mlir::LogicalResult emitDeferredWideEpilogue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::StandaloneReduceOp reduce,
      tcrvrvv::DequantizeOp dequant, tcrvrvv::StoreOp storeOp, mlir::Value accVar,
      tcrvrvv::VectorType accVecType, tcrvrvv::VectorType reduceVecType,
      mlir::Value accVlmax, mlir::Value accBuffer,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the 2nd-family (i16 dot-reduce) deferred-wide body as a real RVV
  /// strip loop (the measured ssh-rvv winner dot_wide_deferred):
  ///   size_t vlmax_acc = __riscv_vsetvlmax_e32m8();
  ///   vint32m8_t dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vlmax_acc);
  ///   for (i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e16m4(n - i);
  ///     vint16m4_t a = __riscv_vle16_v_i16m4(lhs + i, vl);
  ///     vint16m4_t b = __riscv_vle16_v_i16m4(rhs + i, vl);
  ///     vint32m8_t p = __riscv_vwmul_vv_i32m8(a, b, vl);   // SINGLE widening
  ///     dot_acc_vec = __riscv_vadd_vv_i32m8(dot_acc_vec, p, vl); // NON-widening
  ///   }
  ///   // ONE trailing vredsum + scalar acc[0] add + i32 lane-0 store.
  /// The structural marker is tcrv_rvv.deferred_accumulate (I5).
  mlir::LogicalResult emitDeferredWideDotReduceBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The deferred-wide dot-reduce trailing reduction + scalar acc[0] add + i32
  /// lane-0 store (NO dequant):
  ///   vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vsetvlmax_e32m1());
  ///   vint32m1_t vred  = __riscv_vredsum_vs_i32m8_i32m1(dot_acc_vec, vzero, vlmax_acc);
  ///   int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  ///   out[0] = vse32(__riscv_vmv_v_x_i32m1(sum, 1));
  mlir::LogicalResult emitDeferredWideDotReduceEpilogue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::StandaloneReduceOp reduce, tcrvrvv::StoreOp storeOp,
      mlir::Value accVar, tcrvrvv::VectorType accVecType,
      tcrvrvv::VectorType reduceVecType, mlir::Value accVlmax,
      mlir::Value accBuffer, mlir::Value outBuffer,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// True iff `scope` is the STANDALONE i32->f32 runtime-scale dequant body: a
  /// with_vl scope whose ONLY compute is a single load -> dequantize -> store
  /// (no product / reduce / accumulator / clamp). Structurally: exactly one
  /// tcrv_rvv.load producing an i32 vector, exactly one tcrv_rvv.dequantize
  /// (kind `i32_to_f32_scaled`, i32->f32) sourcing that load, exactly one
  /// tcrv_rvv.store of the f32 result, and nothing else. This is the dequant
  /// shape the legacy string materializer Gearbox-unrolls (u2) into a two-slice
  /// runtime-avl setvl loop; `emitStandaloneDequantBody` reproduces that loop.
  /// Any extra op, a different dtype/kind, or a missing piece returns false so
  /// the body falls to the (guarded) emitScopeForLoop path and then the legacy
  /// materializer -- no mislower.
  static bool isStandaloneDequantBody(tcrvrvv::WithVLOp scope);

  /// Emit the standalone i32->f32 runtime-scale dequant body as the legacy
  /// Gearbox-unrolled (u<unroll>) two-slice runtime-avl setvl loop, byte-
  /// identical to the legacy string materializer:
  ///   v5 = setvl(n);  v6 = v5 * unroll;
  ///   for (i = 0; i < n; i += v6) {
  ///     <slice 0>  setvl(n - i)               load(base+i) -> dequant -> store
  ///     <slice 1>  setvl((n - i) - vl0)        load(base+i+vl0) -> ... -> store
  ///     ...
  ///   }
  /// Each slice k recomputes the remaining AVL FRESH from (n - i) and subtracts
  /// the runtime VLs of the prior slices in this iteration; the pointer offset
  /// for slice k>0 is the running sum of prior slice VLs (one extra pointer add
  /// per accumulated VL, matching the legacy `v18=base+i; v19=v18+vl0` form).
  /// The two-slice remaining-VL setvl covers the tail naturally -- there is NO
  /// separate scalar tail loop (unlike the product-reduce dequant routine).
  /// The unroll factor is read from the realized scope's `tcrv_rvv.gearbox.unroll`
  /// attribute (the Gearbox schedule fact); absent or non-positive fails the
  /// match so the body falls back to the legacy materializer unchanged.
  mlir::LogicalResult emitStandaloneDequantBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg,
      mlir::Value vlmax, mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the dequant epilogue once: extract the accumulator's lane-0 scalar
  /// (vmv_x_s), convert to f32, multiply by the runtime scale, then either store
  /// the scalar f32 (the plain dequant), or run the f32 clamp (splat/compare/
  /// select against the lower/upper bounds) before storing. The accumulator is
  /// read from the function-scoped variable, not any in-loop SSA value.
  mlir::LogicalResult emitDequantEpilogue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::DequantizeOp dequant,
      tcrvrvv::StoreOp storeOp, llvm::ArrayRef<mlir::Operation *> epilogueOps,
      mlir::Value accVar, tcrvrvv::VectorType accVecType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

private:
  /// True iff the with_vl body is the pure masked unit-store shape: its only
  /// store-like op is exactly one tcrv_rvv.masked_store, and it contains NO
  /// compute op (binary/macc/compare/select/reduce/dequantize/...) that would
  /// require an agnostic or `_tu`/`_tum` intrinsic form the converter does not
  /// model under undisturbed policy. Such a body is the masked-store family
  /// (mask_load + payload load + masked_store) whose undisturbed scope policy is
  /// honored by the masked-store `_m` intrinsic. Any other shape (a plain store,
  /// a compute op, an extra store) is NOT this exception and stays refused.
  static bool isPureMaskedStoreBody(tcrvrvv::WithVLOp scope);

  /// True iff the with_vl body is the runtime-scalar computed-mask masked-store
  /// shape: its only store-like op is exactly one tcrv_rvv.masked_store whose
  /// predicate is produced by a tcrv_rvv.compare in the same scope, and whose
  /// compare RHS is a tcrv_rvv.splat of a runtime scalar (load -> splat ->
  /// compare -> masked_store, the RuntimeScalarComputedMaskStore family). That
  /// body carries an undisturbed scope policy honored by the masked-store `_m`
  /// form: the compare/splat/load steps emit agnostic intrinsics whose results
  /// are fully defined over the active VL, so the undisturbed semantics live
  /// entirely in the `_m` store -- correctly lowered.
  ///
  /// The runtime-scalar (splat) compare RHS is REQUIRED: it is the only
  /// legitimate unit store-only computed-mask family. A unit masked_store fed by
  /// a VECTOR-vector compare with no load-merge and no splat is NOT a real
  /// family (its closest sibling, ComputedMaskUnitLoadStore, is a load-merge);
  /// such a body must fall back so the legacy validator rejects it (the
  /// stage2-masked-store-negative contract). Any other compute op, plain/
  /// strided/indexed store, masked load, or a second store likewise drops the
  /// body out of this bounded exception so nothing is mislowered under
  /// undisturbed policy.
  static bool isComputedMaskMaskedStoreBody(tcrvrvv::WithVLOp scope);

  /// Capability config gate (I1-honoring). The selected variant's `requires`
  /// symbols resolve to tcrv.exec.capability / tcrv.exec.target provider ops in
  /// the kernel; those are queryable MLIR objects that may declare
  /// `supported_sew` / `supported_lmul` as a comma-separated allow-list. If a
  /// resolved provider declares one of these and it does NOT include the typed
  /// body's (sew, lmul), the capability gates this body out: fail the match so
  /// the body falls back to the legacy validator (which rejects it with the
  /// "supported_sew fact ... does not include typed body SEW" diagnostic).
  /// Reading the attrs straight off the provider op keeps the capability the
  /// legality authority -- no string capability model is imported. The gate is
  /// silent when a provider declares no restriction (the common case).
  ///
  /// `bodyRequiresAgnosticPolicy` carries the typed body's ISA-generation
  /// requirement: a tail/mask-agnostic (ta/ma) policy is a RATIFIED RVV1.0
  /// feature absent on RVV0.7 (xtheadvector / C920). If the resolved provider
  /// declares `rvv_version` = "0.7" and the body requires the agnostic policy,
  /// the capability gates this body out the same way -- this is the N1
  /// ISA-generation divergence, gated on the version CAPABILITY FACT (I3: no
  /// family-name / march-string branch). The gate is silent when the provider
  /// declares no `rvv_version` or declares "1.0".
  mlir::LogicalResult
  checkCapabilityConfigGate(mlir::ConversionPatternRewriter &rewriter,
                            tcrv::exec::VariantOp variant,
                            tcrv::exec::KernelOp kernel, unsigned bodySEW,
                            llvm::StringRef bodyLMUL,
                            bool bodyRequiresAgnosticPolicy) const;

  /// load(%abi, %vl) -> ptr = base + i; __riscv_vle<sew>_v_<dtype><lmul>(ptr, vl)
  /// When `extraOffset` is set, a SECOND pointer add is emitted after the
  /// `base + i` add: ptr2 = ptr + extraOffset (mirroring the legacy unrolled
  /// `v18 = base + i; v19 = v18 + priorVL` two-add form for the second u2 slice).
  /// Existing single-slice callers pass the default null and emit one add.
  mlir::LogicalResult
  emitLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::LoadOp load,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value inductionVar, mlir::Value bodyVL,
           mlir::Value extraOffset = {}) const;

  /// binary{kind}(%lhs,%rhs,%vl) -> __riscv_v<op>_vv_<dtype><lmul>(lhs,rhs,vl)
  mlir::LogicalResult
  emitBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::BinaryOp binary,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const;

  /// reduce{kind}(%input,%acc,%vl) ->
  ///   __riscv_v<red>_vs_<dtype><lmul>_<dtype>m1(input, acc, vl)
  ///
  /// The generic `reduce` family (operation kind ReduceAdd, memory form
  /// vector-rhs-load) seeds the reduction with the rhs-loaded accumulator VECTOR
  /// (lane 0 holds the running seed for this VL chunk) and writes the lane-0
  /// reduction result straight back to the output chunk base with a VL=1 store.
  /// That per-chunk store is the same `emitStore` path; the VL=1 detail is
  /// handled by `reduceResultStoreVL` below. Here we only emit the reduction
  /// call itself.
  ///
  /// Malformed-body guard: this converter only takes the per-chunk
  /// vector-seeded shape (accumulator typed VECTOR + the chunk-base result
  /// layout). A body whose reduce carries the scalar-carry standalone layout, an
  /// unsupported kind, or an unconvertible (dtype, lmul) is NOT lowered here --
  /// notifyMatchFailure rolls the conversion back so the legacy owner/validators
  /// still see (and reject/own) it.
  mlir::LogicalResult
  emitReduce(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::ReduceOp reduce,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const;

  //===--------------------------------------------------------------------===//
  // Standalone (scalar-carry-through-memory) reduction family.
  //
  // Unlike the per-chunk `reduce` family (which seeds with a freshly loaded
  // accumulator vector and writes the lane-0 result to the chunk base), the
  // standalone reduction carries ONE scalar accumulator through the output
  // memory cell `out[0]` across runtime VL chunks:
  //   PRE-LOOP: out[0] = acc[0]    (splat the seed read from the accumulator
  //                                 buffer into a lane-0 m1 vector, store VL=1).
  //   IN-LOOP : out[0] = reduce(input_chunk, splat(out[0]))   (read the running
  //                                 scalar back, splat it, horizontal-reduce the
  //                                 input chunk over it, store back VL=1).
  // The seed/result vector is ALWAYS lmul m1 (the reduction lands its scalar in
  // lane 0 of an m1 destination); the SOURCE may be a wider/fractional lmul
  // (m2, or the widening mf2 rung). Byte-identical to the legacy
  // RVVEmitCReductionAccumulationStatementPlanOwners.cpp standalone oracle.
  //===--------------------------------------------------------------------===//

  /// True iff `op` is a (masked) standalone reduction carrying the scalar-carry
  /// output layout. This is the discriminator separating the standalone family
  /// from the per-chunk `reduce` family.
  static bool isStandaloneReductionOp(mlir::Operation *op);

  /// True iff the with_vl body is a standalone reduction body (it carries a
  /// scalar-carry standalone_reduce / masked_standalone_reduce).
  static bool isStandaloneReductionBody(tcrvrvv::WithVLOp scope);

  /// The reduction mnemonic for the standalone family, mirroring the legacy
  /// getRVVSelectedBodyStandaloneReductionIntrinsic kind table. The widening
  /// kinds reduce a (fractional-lmul) i16 source into an i32 accumulator via
  /// vwredsum; the plain kinds use vred{sum,min,max}. Unknown kinds fail the
  /// match so the body falls back to the legacy validators unchanged.
  static std::optional<llvm::StringRef>
  standaloneReductionMnemonic(llvm::StringRef kind);

  /// The inactive-lane neutral element a computed-mask reduction merges into
  /// the masked-out source lanes before reducing, mirroring the legacy
  /// getRVVStandaloneReductionStatementPlanInactiveNeutral table:
  ///   add -> "0"; min -> INT_MAX (sew-sized); max -> INT_MIN (sew-sized).
  static std::optional<llvm::StringRef>
  maskedStandaloneReductionNeutral(llvm::StringRef kind, unsigned sew);

  /// Read `buffer[0]` as a scalar then splat it into a lane-0 m1 seed vector via
  /// vmv_v_x with VL literal 1 -- the scalar-carry primitive shared by the
  /// pre-loop seed and the in-loop running-seed read. `resultVecType` is the
  /// reduction RESULT vector type (always m1); its element drives the splat
  /// intrinsic dtype/sew and the `int32_t v = base[0];` temp element. The
  /// provenance comment carries the reduction op's source-op name/role so the
  /// rendered C matches the legacy `vmv_v_x` step. Returns the splat result
  /// Value, or nullptr on an unconvertible/typeless shape (caller fails match).
  mlir::Value emitScalarSeedSplat(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value buffer,
                                  tcrvrvv::VectorType resultVecType,
                                  llvm::StringRef sourceOpName,
                                  llvm::StringRef sourceRole) const;

  static mlir::Type getSizeType(mlir::ConversionPatternRewriter &rewriter);

  /// Emit the pre-loop seed of a standalone reduction: out[0] = acc[0]. Reads
  /// the accumulator-seed buffer's first element, splats it into a lane-0 m1
  /// vector, and stores it to the output buffer BASE with VL=1. Runs between the
  /// pre-loop full-chunk setvl and the for-loop. Returns failure (caller falls
  /// back) on a malformed standalone body (missing/mismatched acc/out buffers,
  /// unconvertible result type).
  mlir::LogicalResult emitStandaloneReductionPreLoopSeed(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Store a lane-0 reduction vector to the output buffer BASE (no `+ i`) with
  /// VL literal 1 -- the scalar-output store shared by the pre-loop seed and the
  /// in-loop result. Mirrors the legacy `vse<sew>_v_<dtype>m1(out, v, 1)`.
  mlir::LogicalResult emitStandaloneReductionScalarStore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::StoreOp store, mlir::Value outBuffer, mlir::Value value,
      tcrvrvv::VectorType resultVecType) const;

  /// In-loop plain standalone reduce:
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1);
  ///   v<rd>m1 red = __riscv_v<red>_vs_<src><srcLmul>_<rd>m1(input, seed, vl);
  /// The output buffer is the body's store target (the scalar-carry cell). The
  /// store itself is emitted by the store dispatch (to base, VL=1).
  mlir::LogicalResult
  emitStandaloneReduce(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Location loc, tcrvrvv::StandaloneReduceOp reduce,
                       llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                       mlir::Value outBuffer, mlir::Value bodyVL) const;

  /// In-loop computed-mask standalone reduce:
  ///   v<src><sl> neutral = vmv_v_x_<src><sl>(<NEUTRAL>, vl);  // input lmul
  ///   v<src><sl> masked  = vmerge_vvm_<src><sl>(neutral, source, mask, vl);
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1);
  ///   v<rd>m1 red = __riscv_v<red>_vs_<src><sl>_<rd>m1(masked, seed, vl);
  /// The neutral fills the masked-out lanes so they don't affect the reduction.
  mlir::LogicalResult emitMaskedStandaloneReduce(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedStandaloneReduceOp reduce,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, mlir::Value outBuffer,
      mlir::Value bodyVL) const;

  //===--------------------------------------------------------------------===//
  // Widening contraction family (signed low-precision products / dot-reduce /
  // widening macc). Byte-identical to the legacy
  // RVVEmitCContractionRouteFamilyPlanOwners.cpp DirectContraction oracle.
  //===--------------------------------------------------------------------===//

  /// widening_product(%lhs,%rhs,%vl){kind=signed_widening_product} ->
  ///   v<rd><rl> p = __riscv_vwmul_vv_<rd><rl>(lhs, rhs, vl);
  /// The vwmul widens the narrower (fractional-lmul) source multiplicands into
  /// the RESULT (one-step-wider) vector type, so the intrinsic dtype/lmul come
  /// from the RESULT type (i8/mf4 x i8/mf4 -> i16/mf2 == __riscv_vwmul_vv_i16mf2).
  /// Only the signed i-source slice is in scope; the unsigned (ui) source uses a
  /// vwmulu intrinsic the legacy provider has NOT productionized for this body
  /// converter, so an unsigned source falls back to the legacy path unchanged.
  mlir::LogicalResult
  emitWideningProduct(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, tcrvrvv::WideningProductOp product,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value bodyVL) const;

  /// block_fp16_scale_product(%lhs_base, %rhs_base) lowers the per-block
  /// dual-fp16 SCALE reconstruction to two scalar fp16->fp32 reads (the ONE
  /// sanctioned opaque piece, kFp16ScaleReadCallee) and a scalar float
  /// multiply, byte-identical to the `d_x * d_y` the monolithic block-dot
  /// emitters produce inline:
  ///   float d_x = (float)*(const _Float16 *)(lhs_base);  // emitc.call_opaque
  ///   float d_y = (float)*(const _Float16 *)(rhs_base);  // emitc.call_opaque
  ///   float scale = d_x * d_y;                           // emitc.mul (float)
  /// The optional lhs/rhs scale byte offsets add an emitc.add pointer bump
  /// before the read when non-zero (default 0 -- the fp16 header is the first
  /// AoS byte, matching the monolithic q8_0 read at the block base). The op is
  /// scalar (no vl); the bodyVL argument is unused, taken only to keep the
  /// generic body-walk emitter signature uniform.
  mlir::LogicalResult emitBlockFp16ScaleProduct(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::BlockFp16ScaleProductOp scaleProduct,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// block_computed_scale_dequant(%sumi, %computed_scale) lowers the per-block
  /// computed-scale i32-sumi dequant fold to the SAME scalar i32 -> float
  /// emitc.cast (the `(float)sumi` sitofp) + scalar float emitc.mul the
  /// monolithic block-dot emitters produce inline for the per-block
  /// `(float)sumi * <scale>` term, byte-identical at the operation-spelling
  /// level:
  ///   float term = (float)sumi * scale;   // emitc.cast + emitc.mul (float)
  /// The computed_scale operand is the f32 tcrv_rvv.block_fp16_scale_product
  /// output (M-FLAT brick 1), NOT an imported ABI scale. The op stops at the
  /// per-block term; the cross-block fp32 accumulate is a separate typed step.
  /// The op is scalar (no vl); the bodyVL argument is unused, taken only to
  /// keep the generic body-walk emitter signature uniform.
  mlir::LogicalResult emitBlockComputedScaleDequant(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::BlockComputedScaleDequantOp dequant,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// cross_block_f32_accumulate(%acc, %term) lowers the cross-block fp32
  /// accumulate to the SAME scalar float emitc.add the monolithic block-dot
  /// emitters produce inline for `sumf + <block term>`, byte-identical at the
  /// operation-spelling level:
  ///   float sumf = acc + term;   // emitc.add (float)
  /// acc is the block-carried f32 accumulator, term is the f32
  /// tcrv_rvv.block_computed_scale_dequant output (M-FLAT brick 2). The op is
  /// scalar (no vl); the bodyVL argument is unused, taken only to keep the
  /// generic body-walk emitter signature uniform.
  mlir::LogicalResult emitCrossBlockF32Accumulate(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::CrossBlockF32AccumulateOp accumulate,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// typed_vector_lane0_to_scalar_extract(%red, %vl) lowers the i32 LMUL m1
  /// lane0 -> scalar i32 extract bridge to the SAME __riscv_vmv_x_s_i32m1_i32
  /// call_opaque the monolithic block-dot lowering emits for its vwredsum lane0
  /// -> scalar sumi extraction (RVVToEmitCBlockQuantLinear.cpp:5653-5655),
  /// byte-identical at the operation-spelling level:
  ///   int32_t sumi = __riscv_vmv_x_s_i32m1_i32(red);
  /// The intrinsic targets lane 0 regardless of vl, so the call takes only the
  /// mapped i32m1 vector value; the op's vl operand is the boundary marker (the
  /// bodyVL argument is unused, kept only to match the generic body-walk emitter
  /// signature). Emitting an RVV intrinsic via call_opaque is NOT the monolithic
  /// hand-written block-dot helper (the same shape as vwmul / vwredsum lower).
  mlir::LogicalResult emitTypedVectorLane0ToScalarExtract(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::TypedVectorLane0ToScalarExtractOp extract,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// packed_i4_nibble_unpack_product(%lhs,%rhs,%vl) lowers to the FIXED signed
  /// i4-nibble sign-extend + widening-product intrinsic chain (each i8 packs two
  /// signed 4-bit nibbles), byte-equivalent to the legacy packed-i4 oracle:
  ///   // low nibble: shift the i4 into the high nibble (vsll 4), widen-multiply,
  ///   // arithmetic-shift the i16 product back down (vsra 8) -- this both
  ///   // sign-extends the low nibble and rescales the product.
  ///   v8  lhs_low = vsll_vx_i8mf4(lhs, 4, vl);
  ///   v8  rhs_low = vsll_vx_i8mf4(rhs, 4, vl);
  ///   v16 low_p   = vwmul_vv_i16mf2(lhs_low, rhs_low, vl);
  ///   v16 product = vsra_vx_i16mf2(low_p, 8, vl);
  ///   // high nibble: arithmetic-shift sign-extends the high nibble in place,
  ///   // then a widening multiply-accumulate adds its product.
  ///   v8  lhs_high = vsra_vx_i8mf4(lhs, 4, vl);
  ///   v8  rhs_high = vsra_vx_i8mf4(rhs, 4, vl);
  ///   v16 product  = vwmacc_vv_i16mf2(product, lhs_high, rhs_high, vl);
  /// The op carries the nibble-unpack STRUCTURE typed; the conversion never
  /// reads operand_form / unpack_intent candidate-mirror strings to choose it.
  mlir::LogicalResult emitPackedI4NibbleUnpackProduct(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::PackedI4NibbleUnpackProductOp packed,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// packed_i4_offset_binary_x_i8_product(%weight,%act_low,%act_high,%vl) lowers
  /// to the ASYMMETRIC offset-binary one-sided decode + plain-i8 widening
  /// product/accumulate chain (the integer core of ggml Q4_0 x Q8_0). ONLY the
  /// packed-i4 weight is nibble-decoded; the two int8 activations stay plain:
  ///   // offset-binary -> two's-complement: ggml's (nibble - 8) equals the
  ///   // signed 4-bit value of (nibble XOR 0x8). XOR-ing both nibbles of every
  ///   // byte with 0x88 converts each offset-binary nibble into a
  ///   // two's-complement 4-bit lane, decoded by the existing sign-extend.
  ///   v8  w_xor = vxor_vx_i8mf4(weight, 0x88, vl);    // (cast to int literal)
  ///   // low nibble: shift the i4 into the high nibble (vsll 4), arithmetic
  ///   // shift back (vsra 4) to sign-extend it into a plain [-8,7] i8 lane.
  ///   v8  w_low_sh = vsll_vx_i8mf4(w_xor, 4, vl);
  ///   v8  v0       = vsra_vx_i8mf4(w_low_sh, 4, vl);   // decoded low nibble i8
  ///   // high nibble: arithmetic-shift sign-extends it in place.
  ///   v8  v1 = vsra_vx_i8mf4(w_xor, 4, vl);            // decoded high nibble i8
  ///   // asymmetric widening product: i8(decoded) x i8(plain activation) -> i16.
  ///   v16 product = vwmul_vv_i16mf2(v0, act_low, vl);  // low half  <-> q8[0..15]
  ///   v16 product = vwmacc_vv_i16mf2(product, v1, act_high, vl); // high half
  /// The plain int8 activations are NOT shifted and there is NO vsra(product,8)
  /// rescale -- v0/v1 are real signed i8 lanes, distinct from the symmetric
  /// packed-i4 path which shifts BOTH operands and rescales the product. The op
  /// carries the offset-binary one-sided unpack STRUCTURE typed; the conversion
  /// never reads operand_form / mirror strings to choose it.
  /// Emit the COMPLETE ggml ggml_vec_dot_q4_0_q8_0 block kernel for one
  /// tcrv_rvv.q4_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph):
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);   // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 16; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e<W>m1(16 - c);  // mf4: e32m1 / m1: e8m1
  ///       vint8<L>_t  w  = __riscv_vle8_v_i8<L>(xb + 2 + c, vl);
  ///       vint8<L>_t  y0 = __riscv_vle8_v_i8<L>(yb + 2 + c, vl);
  ///       vint8<L>_t  y1 = __riscv_vle8_v_i8<L>(yb + 2 + 16 + c, vl);
  ///       vint16<W>_t p  = <offset-binary decode/product>(w, y0, y1, vl);
  ///       vint32m1_t  seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t  red  = __riscv_vwredsum_vs_i16<W>_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + ((float)sumi * d_x) * d_y;  // ggml left-assoc order
  ///   }
  ///   *s = sumf;
  /// The strip loop anchors at the integer_core_lmul resource fact: "mf4" (the
  /// INC-2a default: i8mf4 -> i16mf2, vsetvl_e32m1, 4 chunks at VLEN=128) or "m1"
  /// (i8m1 -> i16m2, vsetvl_e8m1, ONE strip + ONE vwredsum per block at VLEN=128,
  /// matching ggml's hand-written reduction anchor). Both are byte-exact (same
  /// integer set; integer add is order-independent) and stay VLEN-robust (a VLEN
  /// < 128 board re-strips correctly via the sumi-carrying seed).
  /// The block format facts (QK/strides/offsets/scale model) are the op's typed
  /// attrs (I4 mirror); the emission is the op's fixed structure. The integer
  /// core reuses emitOffsetBinaryDecodeProductValue (the SAME nodes INC-1
  /// emits). The ONE sanctioned opaque piece is the scalar fp16->fp32 read,
  /// which goes through emitc.call_opaque exactly as INC-1 emits its intrinsics.
  mlir::LogicalResult emitQ4_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the ggml Q4_0 x Q8_0 GEMM tile (weight-decode reuse) for one
  /// tcrv_rvv.q4_0_q8_0_gemm_tile op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// WEIGHT-DECODE-REUSE sibling of emitQ4_0Q8_0BlockDot: ONE weight row times M
  /// activation columns, decoding each q4_0 weight block ONCE and reusing the
  /// decoded v0/v1 nibble lanes across all M columns:
  ///   float sumf[M];  for (j) sumf[j] = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;            // weight block (shared)
  ///     float d_x = (float)*(const _Float16 *)(xb);
  ///     // HOISTED weight decode (m1 whole-half-block, VLEN >= 128):
  ///     size_t vl = __riscv_vsetvl_e8m1(16);
  ///     vint8m1_t w  = __riscv_vle8_v_i8m1(xb + 2, vl);
  ///     vint8m1_t v0, v1 = <offset-binary decode>(w, vl);   // reused M-fold
  ///     for (size_t j = 0; j < M; j += 1) {        // inner column loop
  ///       const uint8_t *yb = vy + j*by + ib*34;
  ///       float d_y = (float)*(const _Float16 *)(yb);
  ///       vint8m1_t y0 = __riscv_vle8_v_i8m1(yb + 2, vl);
  ///       vint8m1_t y1 = __riscv_vle8_v_i8m1(yb + 2 + 16, vl);
  ///       vint16m2_t p = <product from decoded>(v0, v1, y0, y1, vl);
  ///       vint32m1_t seed = __riscv_vmv_v_x_i32m1(0, 1);
  ///       vint32m1_t red  = __riscv_vwredsum_vs_i16m2_i32m1(p, seed, vl);
  ///       int32_t sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///       sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  // ggml order
  ///     }
  ///   }
  ///   for (j) s[j] = sumf[j];
  /// Each column j computes EXACTLY ggml_vec_dot_q4_0_q8_0(weight_row, column_j)
  /// (the weight decode is shared, but the per-column products / reduction / fold
  /// are identical, with M INDEPENDENT fp32 accumulators), so the M outputs are
  /// byte-exact vs M independent vec_dot calls. The weight decode reuses
  /// emitOffsetBinaryDecodeValue (the decode half INC-1 emits); the per-column
  /// product reuses emitOffsetBinaryProductFromDecodedValue (the product half).
  /// The m1 whole-half-block decode is correct at VLEN >= 128 (the board's
  /// mandated full-V floor; vsetvl_e8m1(16) caps the active vl at 16 when
  /// VLMAX >= 16) -- the same justification as the block-dot strip_elision form.
  mlir::LogicalResult emitQ4_0Q8_0GemmTile(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml Q4_0 x Q8_0 FULL GEMM (NR weight rows x nc
  /// activation columns) for one tcrv_rvv.q4_0_q8_0_gemm op as fully STRUCTURED
  /// emitc nodes (I5; ZERO raw() strings -- every value is a node in the IR
  /// graph). It is the FULL-MATMUL wrapper of emitQ4_0Q8_0GemmTile: an outer
  /// emitc.for weight-ROW loop over nr, and an inner emitc.for column-strip loop
  /// over the nc columns in M-wide strips, each strip running G1's
  /// weight-decode-reuse tile body (decode each (row, block) weight ONCE, reuse
  /// the decoded v0/v1 nibble lanes across the M columns of the strip). The
  /// shape is:
  ///   for (size_t ir = 0; ir < nr; ++ir) {
  ///     const uint8_t *xr = vx + ir*bx;      // weight row ir
  ///     float *sr = s + ir*bs;               // output row ir
  ///     for (size_t cb = 0; cb < nc; cb += M) {
  ///       size_t active = (nc - cb < M) ? (nc - cb) : M;  // tail clamp
  ///       float sumf[M]; for (j<active) sumf[j] = 0;
  ///       const uint8_t *yb0 = vy + cb*by;    // first column of the strip
  ///       for (size_t ib = 0; ib < n/QK; ++ib) {
  ///         const uint8_t *xb = xr + ib*18;
  ///         float d_x = (float)*(_Float16*)xb;
  ///         size_t vl = vsetvl_e8m1(16);
  ///         vint8m1_t w = vle8(xb+2, vl);  decode -> v0,v1;   // HOISTED once
  ///         for (size_t j = 0; j < active; ++j) {
  ///           const uint8_t *yb = yb0 + j*by + ib*34;
  ///           float d_y = (float)*(_Float16*)yb;
  ///           y0=vle8(yb+2); y1=vle8(yb+2+16);
  ///           prod = vwmul/vwmacc(v0,y0,v1,y1);
  ///           sumi = vmv_x_s(vwredsum(prod));
  ///           sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  // ggml order
  ///         }
  ///       }
  ///       for (size_t j = 0; j < active; ++j) sr[cb + j] = sumf[j];
  ///     }
  ///   }
  /// Each (ir, cb+j) output is EXACTLY ggml_vec_dot_q4_0_q8_0(weight_row_ir,
  /// column_(cb+j)) -- the weight decode is shared across the strip, but the
  /// per-(block, column) products / reduction / ascending-block-order fp32 fold
  /// are identical to vec_dot, with one INDEPENDENT fp32 accumulator per column,
  /// so every output is byte-exact vs the per-(row,col) vec_dot ggml runs at
  /// VLEN=128 (the repack GEMM disabled at `case 128`). The weight decode reuses
  /// emitOffsetBinaryDecodeValue + emitOffsetBinaryProductFromDecodedValue
  /// (INC-1's decode/product halves) -- the SAME nodes the per-row block dot and
  /// the GEMM tile emit. The m1 whole-half-block decode is correct at VLEN >= 128
  /// (the board's mandated full-V floor).
  mlir::LogicalResult emitQ4_0Q8_0Gemm(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The bounded per-block context the shared q4_0 16x1-REPACKED GEMM lane-wise
  /// integer CORE leaf reads. opName/role are the source-op provenance; l8/l16/l32
  /// are the three element-width rungs of the i8 -> i16 -> i32 widening chain
  /// (mf2 -> m1 -> m2 default, or m1 -> m2 -> m4 whole-LMUL); nibbleBytes is qk/2;
  /// weightInterleave is the 16-way block-as-lane width; weightQuantOffset /
  /// activationQuantOffset locate the repacked nibbles / the q8_0x4 quants within
  /// their blocks; activationInterleave is the 4-way activation-column interleave;
  /// activationHighRow is the high-half quant row (activationInterleave*nibbleBytes
  /// == 64); vl8 is the compile-time-constant strip-width literal. Unlike the GEVM
  /// integer core (which loops numHalves strips over a single activation column)
  /// this GEMM core reads ONE strip (a runtime `roff` value) across the [cLo,cHi)
  /// interleaved activation columns.
  struct RepackGemmQ4IntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    llvm::StringRef l8;
    llvm::StringRef l16;
    llvm::StringRef l32;
    int64_t nibbleBytes;
    int64_t weightInterleave;
    int64_t weightQuantOffset;
    int64_t activationQuantOffset;
    int64_t activationInterleave;
    int64_t activationHighRow;
    mlir::Value vl8;
    mlir::Type sizeType;
  };

  /// The shared q4_0 16x1-REPACKED GEMM per-block LANE-WISE integer CORE leaf:
  /// given the per-block weight base `bl` and activation base `al` (already
  /// advanced by block_index*stride), the runtime strip row offset `roff`
  /// (h*half), and the half-open activation-column range [cLo, cHi), it seeds the
  /// per-column i16 lo/hi accumulators, runs the nibble-step loop (one disjoint
  /// repacked vle8 sub-load at qs[i*16 + roff] + plain sign-extension decode + the
  /// per-column scalar q8_0x4 quant reads + lane-wise vwmacc lo/hi), then combines
  /// lo/hi with a vwadd_vv into the per-column i32 `sumi` values it returns
  /// (indexed by absolute column c, a vector of size activationInterleave filled at
  /// [cLo,cHi)). Factored VERBATIM out of the monolith emitRepackGemmQ4_0Q8_0 so
  /// the monolith AND the typed tcrv_rvv.typed_repack_gemm_loop_body region (a
  /// later milestone) emit byte-identical integer-core C. Emits at the current
  /// insertion point.
  llvm::SmallVector<mlir::Value> emitRepackGemmQ4LaneWiseIntegerCore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const RepackGemmQ4IntegerCoreContext &cx, mlir::Value bl, mlir::Value al,
      mlir::Value roff, int64_t cLo, int64_t cHi) const;

  /// The bounded per-block context the shared q4_0 16x1-REPACKED GEMM per-column
  /// dual-fp16 scale FOLD leaf reads. opName/role are the source-op provenance;
  /// l16/l32 are the f16 scale / f32 fold LMUL rungs; vl8 is the strip-width
  /// literal.
  struct RepackGemmDualFp16ScaleFoldContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    llvm::StringRef l16;
    llvm::StringRef l32;
    mlir::Value vl8;
    mlir::Type sizeType;
  };

  /// The shared q4_0 16x1-REPACKED GEMM per-block per-column dual-fp16 scale FOLD
  /// leaf: given the per-block bases `bl`/`al`, the runtime strip row offset
  /// `roff`, the per-column i32 `sumi` from the integer core, the per-column f32
  /// accumulator lvalues `sumfVar`, and the column range [cLo,cHi), it loads the
  /// per-strip fp16 weight scales once (vle16 at &bl.d[roff]), then per column
  /// reads the single _Float16 activation scale (*(const _Float16 *)&al.d[c]),
  /// widens d = vfwmul(weight_scale, act_scale), converts the column sumi with
  /// vfcvt_f_x_v, and folds acc = vfmacc(acc, sumi_f, d) back into `sumfVar[c]`.
  /// Factored VERBATIM out of the monolith emitRepackGemmQ4_0Q8_0. Emits at the
  /// current insertion point.
  void emitRepackGemmDualFp16ScaleFold(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const RepackGemmDualFp16ScaleFoldContext &cx, mlir::Value bl,
      mlir::Value al, mlir::Value roff, llvm::ArrayRef<mlir::Value> sumi32,
      llvm::ArrayRef<mlir::Value> sumfVar, int64_t cLo, int64_t cHi) const;

  /// The q5_0 block-as-lane sibling of the q4_0 16x1-REPACKED GEVM: q5_0 is q4_0 plus
  /// the 5th high bit. The weight side is block_q5_0x16 (16 interleaved rows, RAW
  /// nibbles at +32, a 64-byte TRANSPOSED bit-packed qh region at +288 carrying
  /// one 16-bit mask per element step, NON-inverted bias bit). Per nibble step
  /// the lane decode expands the qh mask (mask[i] >> h*half, splat, vsrl by lane
  /// id, vand 1 -> qh_bit), assembles the UNSIGNED A = nibble | (qh_bit<<4),
  /// reinterprets u8->i8, then subtracts 16 (vsub) -- the q5_0 offset-binary -16
  /// ((nibble | (qh_bit<<4)) - 16). Activation + dual-fp16 scale fold are
  /// byte-identical to the q4_0 GEMV.
  mlir::LogicalResult emitRepackGemvQ5_0Q8_0(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The option-2 stage-C1b PACK (materialize) emitter: the PRODUCER sibling of
  /// the block-as-lane CONSUMER repack-GEMV/GEMM emitters. It emits ggml's
  /// make_block_q4_0x16 transform (the live blck_size_interleave==1 branch) as a
  /// PURE scalar byte gather + ^0x88 XOR -- NO setvl/with_vl/LMUL/vector ops, NO
  /// fp arithmetic. For each of nblocks output blocks, given the 16 consecutive
  /// source block_q4_0 (stride 18, fp16 d @+0, 16 nibble bytes @+2): copy the 16
  /// fp16 scales VERBATIM into dst d[16] @0..32, then write the 256 interleaved
  /// nibble bytes dst.qs[i] = src[i%16].qs[i/16] ^ 0x88 @32..288 (block_q4_0x16
  /// stride 288). The pack facts (strides/offsets/interleave/xor) are the op's
  /// typed attrs (I4 mirror); the emission is the op's fixed structure (I5,
  /// every value a node). Validated HOST-side byte-exact (memcmp==0) vs ggml's
  /// own inlined make_block_q4_0x16 -- an ISOLATED materialization-capability
  /// proof, e2e-REDUNDANT (ggml packs at load), NEVER a kernel/perf/e2e win.
  mlir::LogicalResult emitPackQ4_0ToX16(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// FAMILY-B (scale+MIN, asymmetric) block-as-lane sibling of
  /// emitRepackGemvQ4_0Q8_0: the q4_1 16x1-REPACKED single-column GEMV (decode).
  /// The weight side is the block_q4_1x16 block-as-lane layout (16 interleaved
  /// rows across 16 vector lanes, dot accumulates LANE-WISE via vwmacc, NO
  /// cross-lane vredsum wall), the activation is ONE plain block_q8_1 stream
  /// (stride 36, quants at +4, scaled-sum s at +2). Two kernel-specific parts
  /// differ from the q4_0 repacked GEMV (the q4_1 quantization is scale+MIN):
  ///   (a) the integer core decodes UNSIGNED nibbles [0,15] (vand 0x0F / vsrl
  ///       0x04 on the u8 weight lane -> reinterpret to i8 -> SAME signed
  ///       lane-wise vwmacc against the plain q8 quants), NOT the offset-binary
  ///       sign-extend chain. The repacked nibbles are stored RAW (no ^0x88);
  ///   (b) the fold carries the per-block MIN correction LANE-WISE: d_x and m_x
  ///       are VECTOR strips (one lane per weight row, vle16 at +0 / +32), d_y
  ///       and s_y are the single activation column's SCALARS (+0 / +2), and the
  ///       fold is ggml's q4_1 statement sumf += (d_x*d_y)*sumi + m_x*s_y folded
  ///       as vfmacc (scale term) then vfadd of vfwmul(m_x, s_y) (MIN term).
  /// Each output is byte-exact vs a scalar/ggml q4_1 reference. The block-format
  /// facts are the op's typed attrs (I4 mirror); the emission is the op's fixed
  /// structure (I5; every value is a node, ZERO raw() strings).
  mlir::LogicalResult emitRepackGemvQ4_1Q8_1(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// FAMILY-B (scale+MIN, asymmetric) block-as-lane PREFILL sibling of
  /// emitRepackGemmQ4_0Q8_0: the q4_1 16x1-REPACKED multi-column GEMM. The weight
  /// side is the block_q4_1x16 block-as-lane layout (16 interleaved rows across 16
  /// vector lanes, dot accumulates LANE-WISE via vwmacc, NO cross-lane vredsum
  /// wall), the activation is the interleaved block_q8_1x4 stream (4 columns;
  /// stride 144, d[4]@+0, s[4]@+8, qs@+16). Two kernel-specific parts differ from
  /// the q4_0 repacked GEMM (the q4_1 quantization is scale+MIN):
  ///   (a) the integer core decodes UNSIGNED nibbles [0,15] (vand 0x0F / vsrl
  ///       0x04 on the u8 weight lane -> reinterpret to i8 -> SAME signed
  ///       lane-wise vwmacc against the plain q8 quants), NOT the offset-binary
  ///       sign-extend chain; the repacked nibbles are stored RAW (no ^0x88);
  ///   (b) the per-column fold carries the per-block MIN correction LANE-WISE:
  ///       d_x/m_x are VECTOR strips (vle16 at +0 / +32), d_y_c/s_y_c are the
  ///       per-column SCALARS, and the fold is ggml's per-(row,col) statement
  ///       sumf += (d_x*d_y_c)*sumi + m_x*s_y_c folded as vfmacc (scale term) then
  ///       vfadd of vfwmul(m_x, s_y_c) (MIN term), once per activation column.
  /// The block-format facts are the op's typed attrs (I4 mirror); the emission is
  /// the op's fixed structure (I5; every value is a node, ZERO raw() strings). The
  /// ABI is the q4_1 extrapolation of ggml's q4_0 repack (ggml ships NO q4_1 repack
  /// GEMM/q8_1x4 quantizer): block_q8_1x4 = {fp16 d[4]; fp16 s[4]; int8 qs[128]}
  /// (stride 144), defined consistent with block_q8_0x4 and the q4_1 GEVM's MIN
  /// strip. Like the q4_1 GEVM sibling this is validated norm-based against a
  /// scalar/in-tree-block-dot reference, NOT byte-exact ggml parity (no upstream
  /// oracle exists); the GEMM's own numeric oracle is deferred this pass.
  mlir::LogicalResult emitRepackGemmQ4_1Q8_1(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The K-quant (super-block) 16x1-REPACKED multi-output-column GEMM (prefill)
  /// emitter: the q4_K sibling of emitRepackGemmQ4_1Q8_1 and the PREFILL sibling
  /// of emitRepackGemvQ4KQ8K. The weight side is the block_q4_Kx16 block-as-lane
  /// layout (16 interleaved columns per lane), the activation the interleaved
  /// block_q8_Kx4 stream (4 rows per group, fp32 d[4]/qs[1024]/bsums[64]). The
  /// q4_K dual d/dmin 8-sub-block 6-bit scale/min unpack is done ONCE per 16
  /// weights and REUSED across the M activation columns (the amortization that is
  /// the prefill e2e-win mechanism). Validated norm-based against the in-tree
  /// q4_K block-dot reference; the byte-exact ggml_gemm_q4_K_16x1_q8_K oracle is
  /// a deferred rvv follow-up.
  mlir::LogicalResult emitRepackGemmQ4KQ8K(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// FAMILY-A (symmetric, full-int8) block-as-lane sibling of
  /// emitRepackGemvQ4_0Q8_0: the q8_0 16x1-REPACKED single-column GEMV (decode).
  /// The weight side is the block_q8_0x16 block-as-lane layout (16 interleaved
  /// rows across 16 vector lanes, dot accumulates LANE-WISE, NO cross-lane
  /// vredsum wall), the activation is ONE plain block_q8_0 stream (stride 34,
  /// quants at +2). Two kernel-specific parts differ from the q4_0 repacked GEMV
  /// (q8_0 weights are FULL int8, not packed nibbles):
  ///   (a) the integer core reads int8 weight lanes DIRECTLY (vle8 -> i8 lane,
  ///       NO vsll/vsra nibble sign-extend, NO ^0x88, NO lo/hi split). Each
  ///       block has 32 contraction positions (not 16 nibble-bytes), one int8
  ///       weight strip per position at qs[i*16 + h*half];
  ///   (b) full int8 products overflow i16 (127*127*3 > 32767), so accumulation
  ///       is i32 IN-BLOCK: vwmul_vx (i8xi8 -> i16 product) then vwadd_wv
  ///       (i32_acc += widened i16 product), NOT q4_0's i16-vwmacc + end-of-block
  ///       vwadd_vv lo/hi combine. The dual-fp16 scale fold (d_x*d_y) is IDENTICAL
  ///       to q4_0. Integer accumulation is order-independent so the dot is
  ///       byte-exact vs a scalar/ggml q8_0 reference.
  /// The block-format facts are the op's typed attrs (I4 mirror); the emission is
  /// the op's fixed structure (I5; every value is a node, ZERO raw() strings).
  mlir::LogicalResult emitRepackGemvQ8_0Q8_0(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml q4_K x q8_K 16x1-REPACKED block-as-lane GEVM for one
  /// tcrv_rvv.repack_gemv_q4_K_q8_K op as fully STRUCTURED emitc nodes (I5; no
  /// raw() strings). The K-quant super-block sibling of emitRepackGemvQ4_1Q8_1:
  /// 8 sub-blocks of 32, per-sub-block 6-bit scale/min unpacked LANE-WISE across
  /// the 16 weight columns, main term d*Sum(scale_sub*sumi_sub) and MIN term
  /// dmin*Sum(min_sub*bsums_sub). Lane-wise accumulator (NO vredsum).
  mlir::LogicalResult emitRepackGemvQ4KQ8K(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The ONE descriptor-driven flat-plain block-dot body (Fork B). It emits the
  /// shared skeleton the q4_0/q8_0/q4_1/q5_0/q5_1 emitters were ~85%-mechanical
  /// mirrors of -- sumf accumulator, nb = n/qk, the block loop (factor-1 or
  /// multi-block unroll main+tail), per-block blockBaseValue address arithmetic,
  /// the per-block fp16 scale reads, the integer core (elided-vs-robust strip),
  /// the strip reduce seed+vwredsum+extract, and the `*s = sumf` store -- and
  /// dispatches the THREE per-op divergence points on descriptor fields: the
  /// decode+product primitive (switch on decodePrimitive over the EXISTING
  /// factored decode helpers), the strip block_len, and the fp32 fold tree
  /// (switch on foldModel). The caller (a thin per-op shim) resolves the ABI
  /// operands + opName/role, derives the descriptor + BlockDotFacts, and calls
  /// this. Byte-identical to the former per-op emitQxxx methods: the emitted
  /// emitc ops are the SAME nodes in the SAME order (the 22/24 block-dot e2e lit
  /// diff core.mlir vs prod.mlir at VLEN128 + VLEN256 proves it).
  mlir::LogicalResult emitFlatBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
      mlir::Value blockDotResult, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, llvm::StringRef opName,
      llvm::StringRef role, const BlockDotFacts &facts,
      const FlatBlockDotDescriptor &descriptor) const;

  /// Build the shared per-block emit state both flat-block-dot callers use. The
  /// interned emitc types + the wideLmul superset formula are single-sourced
  /// here so the monolithic emitFlatBlockDot and the M-FLAT loop-body driver
  /// emit BYTE-IDENTICAL per-block cores. The caller fills sumfVar /
  /// codebookValues after emitting the accumulator decl / table broadcast.
  FlatBlockDotEmitState buildFlatBlockDotEmitState(
      mlir::ConversionPatternRewriter &rewriter,
      const FlatBlockDotDescriptor &descriptor, const BlockDotFacts &facts,
      mlir::Value weightBase, mlir::Value activationBase, mlir::Value sumfVar,
      mlir::Value codebookValues, mlir::Type sizeType, llvm::StringRef opName,
      llvm::StringRef role) const;

  /// The per-block integer core (declares `int32_t sumi = 0`, runs the elided-
  /// or-robust strip decode+product+reduce, returns the sumi lvalue). Factored
  /// VERBATIM from emitFlatBlockDot's former emitIntegerCore lambda (byte-
  /// identical emit); reads all shared state off `st`.
  mlir::FailureOr<mlir::Value>
  emitFlatIntegerCore(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, const FlatBlockDotEmitState &st,
                      mlir::Value xb, mlir::Value yb, mlir::Value qhLow16,
                      mlir::Value qhHigh16, bool forceRobust) const;

  /// The per-block fp32 fold, grouped into ONE emitc.expression (a single C
  /// statement whose fp associativity matches ggml). Factored VERBATIM from
  /// emitFlatBlockDot's former emitFold lambda (byte-identical emit); the fold
  /// tree switches on st.descriptor.foldModel and reads dX/dY (+ mX/sY) as
  /// operands so it is REGION-DRIVEN, not attribute-rederived.
  void emitFlatFold(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, const FlatBlockDotEmitState &st,
                    mlir::Value sumiVar, mlir::Value dX, mlir::Value dY,
                    mlir::Value mX, mlir::Value sY) const;

  /// One full block's integer core + scale reads (addresses + dX/dY + optional
  /// mX/sY + qh + sumi), WITHOUT the fold. Factored VERBATIM from
  /// emitFlatBlockDot's former emitBlockCore lambda (byte-identical emit).
  mlir::FailureOr<FlatBlockCore>
  emitFlatBlockCore(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, const FlatBlockDotEmitState &st,
                    mlir::Value ib, int64_t blockOffset, bool forceRobust) const;

  /// The M-FLAT loop-scaffold emitter (step 1/6): lower the region-carrying
  /// tcrv_rvv.typed_flat_block_dot_loop_body to the byte-exact skeleton
  /// emitFlatBlockDot emits for its mbf==1 form -- the sumf emitc.variable
  /// seeded `0.0f`, nb = n / QK, the outer emitc.for over nb, the SSA
  /// loop-carried acc mapped to a load-at-top / assign-at-bottom of the sumf
  /// lvalue (emitc.for has no iter_args), and the `*s = sumf` scalar store. The
  /// minimal region body (a single tcrv_rvv.cross_block_f32_accumulate over a
  /// stub term) is dispatched through the existing brick emitters; the full
  /// per-block primitive chain + full-body byte-exactness are later steps.
  mlir::LogicalResult emitTypedFlatBlockDotLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The M-FLAT q4_K/q5_K super-block loop-scaffold emitter (milestone-2/3, W5-W7):
  /// lower the region-carrying tcrv_rvv.typed_super_block_block_dot_loop_body to
  /// the byte-exact skeleton the retired q4_K monolith emitted (milestone-3: this
  /// typed emitter is now the SOLE q4_K super-block lowering) -- the function-scoped
  /// aux8[256]/utmp[4]/sums8[8] scratch, the `sums` vfloat32m2 emitc.variable +
  /// `sumf` float emitc.variable DUAL accumulator seeded once OUTSIDE the loop,
  /// nb = n / QK_K, the outer emitc.for over nb, and (post-loop) the sequential
  /// horizontal fold + `*s = sumf` store. The in-loop body is emitted OP-BY-OP
  /// from the region bricks' OPERANDS: each per-super-block base is built from a
  /// brick's (base operand, block_index induction operand) via a shared memo (the
  /// flat W4 blockBaseFor pattern) so the emit provably tracks the region content
  /// (anti-bypass), while the 8 shared CORE helpers (emitQ4_KSuperBlockAux32Core /
  /// MinTermBsumsDot / SumsFoldScaleD / MinTermSubtract / HorizontalFold) keep it
  /// byte-identical to the monolith by construction (same helpers, same facts,
  /// same order). NO monolith retire, NO flip.
  mlir::LogicalResult emitTypedSuperBlockBlockDotLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The M-FLAT q4_0 16x1-REPACKED GEVM loop-scaffold emitter (Phase B, full-body
  /// byte-exact ALL arms): lower the region-carrying
  /// tcrv_rvv.typed_repack_gemv_loop_body to the byte-exact repacked GEVM kernel on
  /// EVERY resource arm -- the VLEN=256 fractional one-strip mf2 form
  /// (numHalves==1, f32m2), the VLEN=128 two-8-lane-halves mf2 form (numHalves==2,
  /// two f32m2), and the RVV0.7 whole-LMUL one-strip m1 form (numHalves==1, f32m4).
  /// nb = n / QK, nc_groups = nc / weight_interleave, the outer emitc.for weight-
  /// column-group loop, the numHalves per-strip vfloat32{m2,m4} emitc.variable
  /// accumulators seeded per group with vfmv_v_f(0.0f), the inner emitc.for block
  /// loop, and the per-strip lane-wise vse32 stores (NO horizontal reduction). The
  /// inner body is FULL-BODY byte-exact: the integer CORE brick
  /// (tcrv_rvv.repack_lane_wise_q4_x_i8_dot -> emitRepackQ4LaneWiseIntegerCore,
  /// numHalves per-strip sumi) FOLLOWED by the numHalves dual-fp16 scale FOLD bricks
  /// (tcrv_rvv.repack_dual_fp16_scale_fold -> ONE call to emitRepackDualFp16Scale
  /// Fold over all strips, which loads each accumulator, folds its sumi in, and
  /// assigns it back). The carried-OUT accumulators are the fold results. The
  /// integer-core LMUL rung (l8/l16/l32) is derived from the loop op's optional
  /// integer_core_lmul (mf2 default, m1 whole-LMUL); every brick is block_index-tied
  /// (anti-bypass) and dataflow-tied fail-closed (I7). Byte-identical to the
  /// monolithic emitRepackGemvQ4_0Q8_0 by construction (shared leaves + same
  /// numHalves/half/l8/l16/l32/byte-offset facts) on every arm.
  mlir::LogicalResult emitTypedRepackGemvLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The M-FLAT q4_0 16x1-REPACKED GEMM loop-scaffold emitter: lower the
  /// region-carrying tcrv_rvv.typed_repack_gemm_loop_body to the byte-exact
  /// repacked GEMM kernel by wrapping the region's inner contraction-block loop in
  /// the SAME four outer loops the now-retired monolithic emitRepackGemmQ4_0Q8_0
  /// emitted -- the
  /// activation-ROW-group (M-tiling) loop over nr/activation_interleave, the
  /// weight-column-group loop over nc/weight_interleave, the RUNTIME strip loop
  /// over numHalves (roff = h*half_lanes), and the compile-time column-PASS loop
  /// (columnsPerPass of activation_interleave per pass). Per pass it seeds the
  /// columnsPerPass per-column f32{m2,m4} accumulators with vfmv_v_f(0.0f), emits
  /// the inner block loop, and stores each column through s + (y*ai + c)*bs +
  /// x*wi + roff with a lane-wise vse32. The inner body drives the SHARED GEMM
  /// leaves emitRepackGemmQ4LaneWiseIntegerCore (the ONE-strip N-column integer
  /// core, producing columnsPerPass per-column sumi) + emitRepackGemmDualFp16Scale
  /// Fold (the per-column scale fold) from the region's CORE + FOLD bricks, so the
  /// whole kernel body is byte-identical to the retired emitRepackGemmQ4_0Q8_0 by
  /// construction (EMPIRICALLY proven at M2 before retirement)
  /// on every arm (RVV1.0 mf2 columnsPerPass=4 one pass; RVV0.7 m1 columnsPerPass=1
  /// four passes). Every brick is block_index + strip_row_offset tied (anti-bypass)
  /// and dataflow-tied fail-closed (I7).
  mlir::LogicalResult emitTypedRepackGemmLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The bounded per-block context the shared q4_0 16x1-REPACKED lane-wise
  /// integer CORE leaf reads. opName/role are the source-op provenance; l8/l16/l32
  /// are the three element-width rungs of the i8 -> i16 -> i32 widening chain
  /// (mf2 -> m1 -> m2 default, or m1 -> m2 -> m4 whole-LMUL); numHalves is the
  /// disjoint-strip count (16/half_lanes); half is the e16m1 strip width;
  /// nibbleBytes is qk/2; weightInterleave is the 16-way block-as-lane width;
  /// weightQuantOffset / activationQuantOffset locate the repacked nibbles / the
  /// q8_0 quants within their blocks; activationHighRow is the high-half quant row
  /// (== nibbleBytes); vl8 is the compile-time-constant strip-width literal.
  struct RepackQ4IntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    llvm::StringRef l8;
    llvm::StringRef l16;
    llvm::StringRef l32;
    int64_t numHalves;
    int64_t half;
    int64_t nibbleBytes;
    int64_t weightInterleave;
    int64_t weightQuantOffset;
    int64_t activationQuantOffset;
    int64_t activationHighRow;
    mlir::Value vl8;
    mlir::Type sizeType;
  };

  /// The shared q4_0 16x1-REPACKED per-block LANE-WISE integer CORE leaf: given
  /// the per-block weight base `bl` and activation base `al` (already advanced by
  /// block_index*stride), it seeds the per-strip i16 lo/hi accumulators, runs the
  /// nibble-step vwmacc loop (one disjoint repacked vle8 sub-load + plain
  /// sign-extension decode + two scalar q8_0 quant reads + lane-wise vwmacc lo/hi
  /// per strip), then combines lo/hi with a vwadd_vv into the per-strip i32 `sumi`
  /// values it returns (numHalves entries). Factored VERBATIM out of the monolith
  /// emitRepackGemvQ4_0Q8_0 so the monolith AND the typed
  /// tcrv_rvv.typed_repack_gemv_loop_body region emit byte-identical integer-core
  /// C (the dual-fp16 per-strip scale fold that consumes the sumi is the caller's,
  /// deferred in the typed loop scaffold). Emits at the current insertion point.
  llvm::SmallVector<mlir::Value> emitRepackQ4LaneWiseIntegerCore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const RepackQ4IntegerCoreContext &cx, mlir::Value bl,
      mlir::Value al) const;

  /// The bounded per-block context the shared q4_0 16x1-REPACKED per-strip
  /// dual-fp16 scale FOLD leaf reads. opName/role are the source-op provenance;
  /// l16/l32 are the f16 scale / f32 fold LMUL rungs of the widening chain
  /// (m1 -> m2 for the mf2 core, m2 -> m4 for the m1 whole-LMUL core); numHalves
  /// is the disjoint-strip count (16/half_lanes); half is the e16m1 strip width;
  /// weightScaleByteOffset / activationScaleByteOffset locate the fp16 scale bytes
  /// within their blocks (0 / 0 -- the `d` field leads each block); vl8 is the
  /// compile-time-constant strip-width literal.
  struct RepackDualFp16ScaleFoldContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    llvm::StringRef l16;
    llvm::StringRef l32;
    int64_t numHalves;
    int64_t half;
    int64_t weightScaleByteOffset;
    int64_t activationScaleByteOffset;
    mlir::Value vl8;
    mlir::Type sizeType;
  };

  /// The shared q4_0 16x1-REPACKED per-block per-strip dual-fp16 scale FOLD leaf:
  /// given the per-block weight base `bl` and activation base `al` (already
  /// advanced by block_index*stride), the per-strip i32 `sumi` values from the
  /// integer core, and the per-strip f32 accumulator lvalues `sumfVar`, it loads
  /// the per-strip fp16 weight scales (vle16), reads the single per-block _Float16
  /// activation scale once, then per strip widens d = vfwmul(weight_scale,
  /// act_scale), converts the sumi with vfcvt_f_x_v, and folds acc = vfmacc(acc,
  /// sumi_f, d) back into `sumfVar[h]`. Factored VERBATIM out of the monolith
  /// emitRepackGemvQ4_0Q8_0 so the monolith AND the typed
  /// tcrv_rvv.typed_repack_gemv_loop_body region's
  /// tcrv_rvv.repack_dual_fp16_scale_fold brick emit byte-identical scale-fold C.
  /// Emits at the current insertion point.
  void emitRepackDualFp16ScaleFold(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const RepackDualFp16ScaleFoldContext &cx, mlir::Value bl, mlir::Value al,
      llvm::ArrayRef<mlir::Value> sumi,
      llvm::ArrayRef<mlir::Value> sumfVar) const;

  /// The M-FLAT q6_K super-block SINGLE-accumulator loop emitter (milestone-2,
  /// W-D): lower the region-carrying tcrv_rvv.typed_super_block_block_dot_loop_body
  /// whose fold_model is "scales_times_sumi" (the q6_K no-min path) to the
  /// byte-exact skeleton the retired q6_K monolith emitQ6_KQ8_KBlockDot emitted --
  /// the function-scoped aux8[256]/sums8[8] scratch, the `sums` vfloat32m2
  /// emitc.variable SINGLE accumulator seeded once OUTSIDE the loop (NO sumf, NO
  /// MIN chain), nb = n / QK_K, the outer emitc.for over nb, and (post-loop) the
  /// sequential horizontal add (seeded from a fresh 0.0f literal) + `*s` store. The
  /// in-loop body is emitted OP-BY-OP from the region bricks' OPERANDS: the q6_K
  /// aux32 integer-core brick (reusing emitQ6_KSuperBlockAux32Core) + the reused
  /// no-min positive-fold brick, each per-super-block base built from its (base,
  /// block_index) via a shared memo (anti-bypass), byte-identical to the monolith
  /// by construction. Dispatched from emitTypedSuperBlockBlockDotLoopBody on the
  /// fold_model. This is the SOLE q6_K super-block lowering.
  mlir::LogicalResult emitTypedSuperBlockScalesTimesSumiLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The M-FLAT q2_K super-block SCALAR-accumulator loop emitter (milestone-2,
  /// W-D): lower the region-carrying tcrv_rvv.typed_super_block_block_dot_loop_body
  /// whose fold_model is "scalar_scale_min" (the q2_K path) to the byte-exact
  /// skeleton the retired q2_K monolith emitQ2_KQ8_KBlockDot emitted -- the
  /// function-scoped aux8[256] scratch, the `sumf` float emitc.variable SCALAR
  /// accumulator seeded once OUTSIDE the loop (NO 8-lane `sums` vector, NO deferred
  /// vector fold, NO post-loop horizontal add), nb = n / QK_K, the outer emitc.for
  /// over nb, and (post-loop) the `*s` store. The in-loop body is emitted OP-BY-OP
  /// from the region's q2_K integer-core brick OPERANDS: the shared q2_K integer
  /// core (emitQ2_KSuperBlockIntegerCore: the 2-bit unpack + plain uint4-nibble
  /// scale/min + per-sub-block scalar i32 dot producing the two SCALAR states isum
  /// + summs) followed by the shared scalar fold (emitQ2_KScalarFold: sumf +=
  /// dall*isum - dmin*summs, fp16 d@80 / dmin@82), the per-super-block base built
  /// from the brick's (base, block_index) via a shared memo (anti-bypass),
  /// byte-identical to the monolith by construction. Dispatched from
  /// emitTypedSuperBlockBlockDotLoopBody on the fold_model. This is the SOLE q2_K
  /// super-block lowering.
  mlir::LogicalResult emitTypedSuperBlockScalarScaleMinLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The M-FLAT iq1_s super-block SCALAR-accumulator GRID loop emitter (the flip
  /// lowering, M3): lower the region-carrying tcrv_rvv.typed_super_block_block_dot_loop_body
  /// whose fold_model is "scalar_delta_grid" (the iq1_s ternary-grid path) to the
  /// byte-exact skeleton the (now-retired) monolith emitIQ1SQ8KBlockDot emitted --
  /// the function-scoped `static const uint64_t tcrv_iq1s_grid[2048]` TERNARY grid
  /// decl, the `sumf` float emitc.variable SCALAR accumulator seeded once OUTSIDE the
  /// loop (NO 8-lane `sums` vector), nb = n / QK_K, the `tcrv_iq1s_grid` base literal,
  /// the outer emitc.for over nb, and (post-loop) the `*s` store. The in-loop body is
  /// emitted OP-BY-OP from the region's iq1_s grid-core brick OPERANDS: the shared
  /// per-super-block grid body (emitIQ1SSuperBlockGridBody: the fp16*fp32 d fold, the
  /// 11-bit grid-index build from qs+qh, the vluxei16 ternary-grid gather, the signed
  /// widening grid dot into the SCALAR sumi + the delta-bsum SCALAR sumi1, then the
  /// scalar fold sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)), the
  /// per-super-block base built from the brick's (base, block_index) via a shared memo
  /// (anti-bypass), byte-identical to the retired monolith by construction (same grid
  /// decl, same body helper, same facts, same order). Dispatched from
  /// emitTypedSuperBlockBlockDotLoopBody on the fold_model. THE FLIP (M3): the front
  /// door now constructs this typed body as the SOLE representation of the iq1_s
  /// vec_dot (the monolith op + emitter + verifier were retired the same action).
  mlir::LogicalResult emitTypedSuperBlockScalarDeltaGridLoopBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The M-FLAT iq1_m super-block SCALAR-accumulator GRID loop emitter (the flip
  /// lowering, iq1_s SIBLING): the iq1_m branch of the fold_model "scalar_delta_grid"
  /// path (dispatched by emitTypedSuperBlockScalarDeltaGridLoopBody when the region
  /// carries an iq1_m grid-core brick). It emits the SAME wrapper as the iq1_s emitter
  /// (the `static const uint64_t tcrv_iq1m_grid[2048]` TERNARY grid decl from the
  /// canonical kIQ1MGrid, the `sumf` float SCALAR accumulator seeded once OUTSIDE the
  /// loop, nb = n / QK_K, the `tcrv_iq1m_grid` base literal, the outer emitc.for over
  /// nb, the per-super-block base built from the brick's (base, block_index) via a
  /// shared memo (anti-bypass), and the `*s` store), delegating the in-loop
  /// per-super-block body to the shared emitIQ1MSuperBlockGridBody anchor (the packed
  /// iq1m_scale fp16 reconstruct + fp32 d fold, the half-split per-half vluxei16 grid
  /// dot with two half scales, the per-group four-sign Σq8 delta, then the scalar delta
  /// fold sumf += d*((float)sumi1 + IQ1M_DELTA*(float)sumi2)). Byte-identical to the
  /// retired monolith emitIQ1MQ8KBlockDot by construction (same grid decl, same body
  /// helper, same facts, same order) modulo the source-op provenance token + func name.
  mlir::LogicalResult emitTypedSuperBlockScalarDeltaGridLoopBodyIq1M(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The iq3_xxs super-block SCALAR-accumulator GRID-of-4 loop emitter (the flip
  /// lowering, iq1_s grid SIBLING): the iq3_xxs branch of the fold_model
  /// "scalar_delta_grid" path (dispatched by emitTypedSuperBlockScalarDeltaGridLoopBody
  /// when the region carries an iq3_xxs grid-core brick). It emits the wrapper -- the
  /// `static const uint32_t tcrv_iq3xxs_grid[256]` GRID-of-4 decl (from the canonical
  /// kIQ3XXSGrid), the `static const uint8_t tcrv_iq3xxs_ksigns[128]` SIGN plane (from
  /// kIQ3XXSKsigns), the inline `tcrv_iq3xxs_kmask[8]` decl, the `sumf` float SCALAR
  /// accumulator seeded once OUTSIDE the loop, nb = n / QK_K, the ONCE 8-lane kmask
  /// load + the (const int32_t *) grid32 view, the outer emitc.for over nb, the
  /// per-super-block base built from the brick's (base, block_index) via a shared memo
  /// (anti-bypass), and the trailing `*s = 0.25f*sumf` store -- delegating the in-loop
  /// per-super-block body to the shared emitIQ3XXSSuperBlockGridBody anchor (the fp16*fp32
  /// d fold, the aux32 4-bit-scale + 4-sign-group ksigns decode, the two-index-per-group
  /// vluxei16_v_i32m1 grid-of-4 gather + signed widening dot + bsum fold, then
  /// sumf += d*(float)bsum). Byte-identical to the retired monolith emitIQ3XXSQ8KBlockDot
  /// by construction (same decls, same body helper, same facts, same order) modulo the
  /// source-op provenance token + func name.
  mlir::LogicalResult emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The iq2_xxs super-block SCALAR-accumulator GRID-of-8 loop emitter (the flip
  /// lowering, iq1_s grid SIBLING, SIGN-PLANE signs64 variant): the iq2_xxs branch of the
  /// fold_model "scalar_delta_grid" path (dispatched by
  /// emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an iq2_xxs
  /// grid-core brick). It emits the wrapper -- the `static const int64_t
  /// tcrv_iq2xxs_grid[256]` GRID-of-8 decl (from the canonical kIQ2XXSGrid), the DERIVED
  /// `static const int8_t tcrv_iq2xxs_signs64[1024]` signs64 SIGN plane (from
  /// kIQ2XXSKsigns), the `sumf` float SCALAR accumulator seeded once OUTSIDE the loop, nb
  /// = n / QK_K, the ONCE (const int64_t *) grid64 + signs64 views, the outer emitc.for
  /// over nb, the per-super-block base built from the brick's (base, block_index) via a
  /// shared memo (anti-bypass), and the trailing `*s = 0.125f*sumf` store -- delegating
  /// the in-loop per-super-block body to the shared emitIQ2XXSSuperBlockGridBody anchor
  /// (the fp16*fp32 d fold, the aux1 4-bit-scale + 4-sign-group decode, the TWO
  /// vluxei16_v_i64<core> grid64/signs64 gathers + the vmul-onto-grid sign fold + signed
  /// widening dot + bsum fold, then sumf += d*(float)bsum). The Win-A m2/m1 gearbox is
  /// preserved (coreLmul off the brick's integer_core_lmul). Byte-identical to the retired
  /// monolith emitIQ2XXSQ8KBlockDot by construction (same decls, same body helper, same
  /// facts, same order) modulo the source-op provenance token + func name.
  mlir::LogicalResult emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The structured E8M0 -> fp32 HALF weight scale (the mxfp4 FP4-class scale
  /// source, FlatWeightScaleSource::E8M0): GGML_E8M0_TO_FP32_HALF(e) = 2^(e-128),
  /// reconstructed from the single shared-exponent byte at `xb` by ggml's EXACT
  /// bit construction (no scalbnf/ldexpf) -- read e, build uint32_t
  /// bits = (e < 2) ? (0x00200000u << (e & 0x1F)) : ((e - 1) << 23), reinterpret
  /// as float via a `*(const float *)&bits` pun. All structured emitc nodes; the
  /// emitted C is byte-identical to the former inline mxfp4 emitter. Returns the
  /// plain-`float` scale the shared fold multiplies as `d_x`.
  mlir::Value emitE8M0HalfScale(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value xb,
                                llvm::StringRef opName,
                                llvm::StringRef role) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq4_xs_q8_K super-block dot-product for
  /// one tcrv_rvv.iq4_xs_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// raw()). It is the CODEBOOK class's SUPER-BLOCK rung -- the super-block variant
  /// of iq4_nl -- composing iq4_nl's codebook-gather integer core (REUSED VERBATIM:
  /// the same broadcast kvalues_iq4nl[16] table + vand 0x0F / vsrl 0x04 -> vrgather
  /// -> the SAME emitOffsetBinaryProductFromDecodedValue product + vwredsum) with
  /// the q4_K-style super-block structure. The genuinely-new piece is the iq4_xs
  /// SIGNED 6-bit per-sub-block scale applied in the FLOAT domain (NOT q6_K's
  /// integer aux32 domain -- this is the byte-exactness pivot at -ffp-contract=off:
  /// `_generic` computes `d1 = d4d8*(ls-32)` then `sumf += d1*sumi`, two separate
  /// float roundings, so the scale must NOT be folded into an integer accumulator):
  ///   static const int8_t tcrv_iq4_xs_kvalues[16] = { ... };  // codebook decl
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   vint8m1_t values = __riscv_vle8_v_i8m1(tcrv_iq4_xs_kvalues, 16);  // ONCE
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*136;  const uint8_t *yb = vy + ibl*292;
  ///     float dx   = (float)*(const _Float16 *)(xb + 0);     // fp16 weight d
  ///     float dy   = *(const float *)(yb + 0);               // fp32 q8_K d
  ///     float d4d8 = dx * dy;                                 // ONCE per super-block
  ///     uint16_t scales_h = *(const uint16_t *)(xb + 2);
  ///     for (size_t j = 0; j < 8; ++j) {                      // FLAT sub-block loop
  ///       // SIGNED 6-bit scale (closed form == ggml's progressive h>>=4):
  ///       uint8_t sl = xb[4 + j/2];
  ///       int ls = ((sl >> (4*(j%2))) & 0xf) | (((scales_h >> (2*j)) & 0x3) << 4);
  ///       float d1 = d4d8 * (float)(ls - 32);                 // SEPARATE op (no fma)
  ///       // iq4_nl codebook core over qs+j*16 (16 nibble bytes) x q8+j*32:
  ///       int32_t sumi = 0;  __riscv_vsetvl_e8m1(16);
  ///       ... vand/vsrl -> vrgather x2 -> vwmul low + vwmacc high -> vwredsum ...
  ///       sumf = sumf + d1 * (float)sumi;                     // ONE emitc.expression
  ///     }
  ///   }
  ///   *s = sumf;
  /// The codebook gather pins the m1 anchor (VLMAX >= 16). The fold is invoked in
  /// STRICT ascending (super-block, sub-block) order so fp non-associativity is
  /// byte-exact across all -ffp-contract modes. A SEPARATE emitter from
  /// emitIQ4NLQ8_0BlockDot (the inner codebook logic is COPIED, not factored, so
  /// iq4_nl's emitted bytes stay byte-identical -- additive).
  mlir::LogicalResult emitIQ4XSQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  // NOTE: the monolith emitIQ2XXSQ8KBlockDot emitter was RETIRED at the iq2_xxs flip
  // (L3 coverage): the front door now constructs the typed super-block
  // SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 66),
  // lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs, which reuses the SHARED
  // byte-exact anchors below (emitIQ2XXSCanonicalGridTableDecl +
  // emitIQ2XXSCanonicalSigns64TableDecl + emitIQ2XXSSuperBlockGridBody).

  /// The bounded per-super-block byte-exact facts the iq2_xxs GRID-of-8 body helper
  /// reads (the I4 mirror off the grid-core brick). The emitc element/pointer types are
  /// re-derived inside the helper from the MLIRContext (uniqued -> the SAME Type
  /// instances), so this struct carries only the provenance, the size type, the two
  /// buffer pointer types, the iq2_xxs format byte offsets/sub-block shape, the Win-A
  /// coreLmul gearbox anchor, and the two per-loop-invariant SSA views the sub-block
  /// gathers read: the `gridName` (const int64_t *) view of tcrv_iq2xxs_grid and the
  /// `signs64` (const int64_t *) view of tcrv_iq2xxs_signs64 (the DERIVED keven_signs
  /// sign plane).
  struct IQ2XXSGridBodyContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t weightDOffset;        //   0 (fp16 x.d)
    int64_t qsOffset;             //   2 (64 uint8 INTERLEAVED index+aux bytes)
    int64_t activationDOffset;    //   0 (fp32 y.d)
    int64_t q8Offset;             //   4 (q8_K quants)
    int64_t subBlock;             //  32
    int64_t numSubBlocks;         //   8
    int64_t numGroups;            //   4 (grid/sign groups per sub-block)
    llvm::StringRef coreLmul;     // the Win-A gearbox anchor ("m2" default / "m1" @VLEN256)
    mlir::Value gridName;         // the (const int64_t *) view of tcrv_iq2xxs_grid
    mlir::Value signs64;          // the (const int64_t *) view of tcrv_iq2xxs_signs64
  };

  /// Emit the fixed 256-entry iq2_xxs GRID-of-8 codebook as ONE `static const int64_t
  /// tcrv_iq2xxs_grid[256] = { ... };` verbatim decl (ggml's exact uint64 hex literals
  /// rendered `0x%016llxULL`), from the CANONICAL kIQ2XXSGrid constant (the grid-core
  /// brick carries no grid in the IR -- the emitter keys the fixed codebook off the brick
  /// op identity). Byte-identical to the retired monolith decl.
  void emitIQ2XXSCanonicalGridTableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit the DERIVED keven_signs_q2xs signs64 SIGN plane as ONE `static const int8_t
  /// tcrv_iq2xxs_signs64[1024] = { ... };` verbatim decl -- the 128-entry ksigns_iq2xs
  /// selector (canonical kIQ2XXSKsigns) EXPANDED to per-lane +-1 (byte b of selector j is
  /// `(ksigns[j] & (1<<b)) ? -1 : +1`). This IS the signs64 sign-plane mechanism carried
  /// by op identity (NO op-attr extension): the retired monolith derived it identically
  /// from its carried ksigns attr, so the decl is byte-identical.
  void emitIQ2XXSCanonicalSigns64TableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit ONE iq2_xxs super-block's GRID-of-8 body at the current insertion point
  /// (INSIDE an already-open super-block loop whose per-super-block bases xb/yb are
  /// provided): the per-super-block `d = fp16(x.d)*y.d` fold scale, the qs/q8 base setup,
  /// the `int32_t bsum = 0`, the flat 8-sub-block decode (the aux1 4-bit scale ls, the 4
  /// grid indices + 4 sign selectors packed as u16 byte-offsets, the TWO
  /// vluxei16_v_i64<core> grid64/signs64 gathers + reinterpret i8<core> + the
  /// vmul_vv_i8<core> sign-onto-grid fold + vwmul_i16<wide> + ONE vwredsum per sub-block,
  /// then `bsum += sumi*ls`), and the per-super-block fp32 fold `sumf += d*(float)bsum`
  /// folded into the carried `sumf` lvalue. The trailing `*s = 0.125f*sumf` stays in the
  /// wrapper (OUT of this body). This is the byte-exact anchor kept across the iq2_xxs
  /// flip: the retired monolith emitIQ2XXSQ8KBlockDot's inline body was code-moved here
  /// (now the sole caller is the typed super-block SCALAR grid loop lowering) -- same
  /// nodes, same order. The coreLmul carries the Win-A m2/m1 gearbox.
  void emitIQ2XXSSuperBlockGridBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const IQ2XXSGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  // NOTE: the monolith emitIQ3XXSQ8KBlockDot emitter was RETIRED at the iq3_xxs flip
  // (L3 coverage): the front door now constructs the typed super-block
  // SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 98),
  // lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs, which reuses the SHARED
  // byte-exact anchors below (emitIQ3XXSCanonicalGridTableDecl +
  // emitIQ3XXSCanonicalKsignsTableDecl + emitIQ3XXSSuperBlockGridBody).

  /// The bounded per-super-block byte-exact facts the iq3_xxs GRID-of-4 body helper
  /// reads (the I4 mirror off the grid-core brick). The emitc element/pointer types are
  /// re-derived inside the helper from the MLIRContext (uniqued -> the SAME Type
  /// instances), so this struct carries only the provenance, the size type, the two
  /// buffer pointer types, the iq3_xxs format byte offsets/sub-block shape, and the two
  /// per-loop-invariant SSA values the sub-block gather reads: the `grid32` (const
  /// int32_t *) view of tcrv_iq3xxs_grid and the 8-lane `kmask` vuint8m1_t.
  struct IQ3XXSGridBodyContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t weightDOffset;        //   0 (fp16 x.d)
    int64_t qsOffset;             //   2 (64 uint8 grid-index bytes)
    int64_t gasOffset;            //  66 (32 aux bytes, one uint32 / sub-block)
    int64_t activationDOffset;    //   0 (fp32 y.d)
    int64_t q8Offset;             //   4 (q8_K quants)
    int64_t subBlock;             //  32
    int64_t numSubBlocks;         //   8
    int64_t numGroups;            //   4 (sign groups per sub-block)
    int64_t indicesPerSubBlock;   //   8 (grid index bytes per sub-block)
    mlir::Value grid32;           // the (const int32_t *) view of tcrv_iq3xxs_grid
    mlir::Value kmask;            // the 8-lane vuint8m1_t {1,2,4,8,16,32,64,128}
  };

  /// Emit the fixed 256-entry iq3_xxs GRID-of-4 codebook as ONE `static const uint32_t
  /// tcrv_iq3xxs_grid[256] = { ... };` verbatim decl (ggml's exact hex literals rendered
  /// `0x%08xU`). The byte-exact SHARED anchor kept across the iq3_xxs flip: the retired
  /// monolith emitIQ3XXSQ8KBlockDot passed its carried grid attr; the typed grid loop
  /// lowering (the sole live caller) passes the canonical kIQ3XXSGrid so the emitted
  /// decl is byte-identical.
  void emitIQ3XXSGridTableDecl(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc,
                               llvm::ArrayRef<int32_t> grid) const;

  /// Emit the iq3_xxs grid decl from the CANONICAL kIQ3XXSGrid constant (the grid-core
  /// brick carries no grid in the IR -- the emitter keys the fixed codebook off the
  /// brick op identity). Byte-identical to the monolith decl whose grid attr is
  /// populated from the same kIQ3XXSGrid.
  void emitIQ3XXSCanonicalGridTableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit the fixed 128-entry ksigns_iq2xs SIGN plane as ONE `static const uint8_t
  /// tcrv_iq3xxs_ksigns[128] = { ... };` verbatim decl, from the canonical kIQ3XXSKsigns
  /// constant (byte-identical to the monolith's carried ksigns attr decl).
  void emitIQ3XXSCanonicalKsignsTableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit ONE iq3_xxs super-block's GRID-of-4 body at the current insertion point
  /// (INSIDE an already-open super-block loop whose per-super-block bases xb/yb are
  /// provided): the per-super-block `d = fp16(x.d)*y.d` fold scale, the q3/gas/q8 base
  /// setup, the `int32_t bsum = 0`, the flat 8-sub-block decode (the aux32 4-bit scale
  /// ls, the 4 sign groups reading ksigns + two grid indices, the vluxei16_v_i32m1
  /// grid-of-4 gather, the signed widening product + ONE vwredsum per group, then
  /// `bsum += sumi*ls`), and the per-super-block fp32 fold `sumf += d*(float)bsum`
  /// folded into the carried `sumf` lvalue. The trailing `*s = 0.25f*sumf` stays in the
  /// wrapper (OUT of this body). This is the byte-exact anchor kept across the iq3_xxs
  /// flip: the retired monolith emitIQ3XXSQ8KBlockDot shared it with the typed
  /// super-block SCALAR-accumulator grid loop lowering (now the sole caller) -- same
  /// nodes, same order.
  void emitIQ3XXSSuperBlockGridBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const IQ3XXSGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq3_s_q8_K super-block dot-product for one
  /// tcrv_rvv.iq3_s_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no raw()).
  /// It is a direct RE-COMPOSITION of three already-built mechanisms, mirroring
  /// _generic (quants.c:1043-1097):
  ///   static const uint32_t tcrv_iq3s_grid[512] = { ... };   // 512-entry GRID-of-4
  ///   static const uint8_t  tcrv_iq3s_kmask[8]  = {1,2,4,8,16,32,64,128};
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   vuint8m1_t kmaskLo = vle8(kmask,4);  vuint8m1_t kmaskHi = vle8(kmask+4,4);
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq3s_grid;
  ///   for (size_t ibl = 0; ibl < nb; ++ibl) {
  ///     const uint8_t *xb = vx + ibl*110;  const uint8_t *yb = vy + ibl*292;
  ///     float d = (float)*(const _Float16 *)(xb) * *(const float *)(yb);
  ///     const uint8_t *qs = xb+2; const uint8_t *qh = xb+66;
  ///     const uint8_t *sgn = xb+74; const uint8_t *sc = xb+106;
  ///     const int8_t *q8 = yb+4;  int32_t bsum = 0;
  ///     for (ib32 = 0..7) {                                 // FLAT unrolled
  ///       int ls = ib32 even ? 2*(sc[ib32/2] & 0xf)+1 : 2*(sc[ib32/2] >> 4)+1;
  ///       int qhb = qh[ib32];  int32_t sumi = 0;
  ///       for (l = 0..3) {
  ///         int signs = sgn[ib32*4 + l];                    // EXPLICIT, from memory
  ///         int idx1 = qs[8*ib32+2l+0] | ((qhb << (8-2l)) & 256);  // grid-of-4 pass A
  ///         int idx2 = qs[8*ib32+2l+1] | ((qhb << (7-2l)) & 256);  // grid-of-4 pass B
  ///         pass(idx1, q8[0..3], kmaskLo); pass(idx2, q8[4..7], kmaskHi); q8 += 8;
  ///       }
  ///       bsum += sumi * ls;
  ///     }
  ///     sumf = sumf + d * (float)bsum;                      // ONE emitc.expression
  ///   }
  ///   *s = sumf;                                            // NO trailing factor
  /// Each grid-of-4 pass: vle8(4) over grid_i8 + idx*4, the SIGN-plane apply (broadcast
  /// signs / vand kmaskLo|kmaskHi / vmsne / vmerge with vneg grid), the signed widening
  /// product vwmul_i16m2, and the chained vwredsum into sumi (order-free integer add).
  /// The grid lookup is an indexed vle8(4) over a pointer (grid_i8 + idx*4); no
  /// vrgather, no m1 table anchor. The per-super-block fp32 fold is invoked in STRICT
  /// ascending order so fp non-associativity is byte-exact across all -ffp-contract
  /// modes. This emitter is iq3_xxs's body (the grid-of-4 two-pass structure) with the
  /// index assembly (qh injection), the sign source (explicit memory), and the scale
  /// source (explicit two-nibble scales[]) swapped to iq2_s's mechanism -- a
  /// self-contained parallel, NOT a shared-helper refactor (siblings stay
  /// byte-identical / the change is additive).
  mlir::LogicalResult emitIQ3SQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The iq2_xs super-block SCALAR-accumulator per-half-scale GRID loop emitter (the flip
  /// lowering, iq2_xxs grid SIBLING, SIGN-PLANE signs64 variant, PER-HALF explicit scale):
  /// the iq2_xs branch of the fold_model "scalar_delta_grid" path (dispatched by
  /// emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an iq2_xs grid-core
  /// brick). It emits the wrapper -- the `static const int64_t tcrv_iq2xs_grid[512]` decl
  /// (from the canonical kIQ2XSGrid), the DERIVED `static const int8_t
  /// tcrv_iq2xs_signs64[1024]` signs64 SIGN plane (from kIQ2XSKsigns), the `sumf` float
  /// SCALAR accumulator seeded once OUTSIDE the loop, nb = n / QK_K, the ONCE
  /// (const int64_t *) grid64 + signs64 views, the outer emitc.for over nb, the
  /// per-super-block base built from the brick's (base, block_index) via a shared memo
  /// (anti-bypass), and the trailing `*s = 0.125f*sumf` store -- delegating the in-loop
  /// per-super-block body to the shared emitIQ2XSSuperBlockGridBody anchor (the fp16*fp32 d
  /// fold, the per-sub-block explicit 4-bit scale ls1/ls2, the two-half 16-lane decode + the
  /// TWO vluxei16_v_i64m1 grid64/signs64 gathers + the vmul-onto-grid sign fold + signed
  /// widening dot + bsum fold, then sumf += d*(float)bsum). UNLIKE iq2_xxs there is NO
  /// gearbox (fixed 16-lane per-half shape). Byte-identical to the retired monolith
  /// emitIQ2XSQ8KBlockDot by construction (same decls, same body helper, same facts, same
  /// order) modulo the source-op provenance token + func name.
  mlir::LogicalResult emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xs(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The bounded per-super-block byte-exact facts the iq2_xs per-half-scale GRID body helper
  /// reads (the I4 mirror off the grid-core brick). The emitc element/pointer types are
  /// re-derived inside the helper from the MLIRContext (uniqued -> the SAME Type instances),
  /// so this struct carries only the provenance, the size type, the two buffer pointer
  /// types, the iq2_xs format byte offsets/sub-block shape, and the two per-loop-invariant
  /// SSA views the sub-block gathers read: the `gridName` (const int64_t *) view of
  /// tcrv_iq2xs_grid and the `signs64` (const int64_t *) view of tcrv_iq2xs_signs64 (the
  /// DERIVED keven_signs sign plane). UNLIKE iq2_xxs there is NO coreLmul (fixed 16-lane
  /// per-half shape); the scalesOffset carries the explicit per-sub-block scale byte stream.
  struct IQ2XSGridBodyContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t weightDOffset;        //   0 (fp16 x.d)
    int64_t qsOffset;             //   2 (64 uint8 qs bytes: 9-bit idx + 7-bit sign sel)
    int64_t scalesOffset;         //  66 (explicit 4-bit scales[8])
    int64_t activationDOffset;    //   0 (fp32 y.d)
    int64_t q8Offset;             //   4 (q8_K quants)
    int64_t subBlock;             //  32
    int64_t numSubBlocks;         //   8
    int64_t numGroupsPerHalf;     //   2 (grid/sign groups per 16-lane half)
    mlir::Value gridName;         // the (const int64_t *) view of tcrv_iq2xs_grid
    mlir::Value signs64;          // the (const int64_t *) view of tcrv_iq2xs_signs64
  };

  /// Emit the fixed 512-entry iq2_xs GRID codebook as ONE `static const int64_t
  /// tcrv_iq2xs_grid[512] = { ... };` verbatim decl (ggml's exact uint64 hex literals
  /// rendered `0x%016llxULL`), from the CANONICAL kIQ2XSGrid constant (the grid-core brick
  /// carries no grid in the IR -- the emitter keys the fixed codebook off the brick op
  /// identity). Byte-identical to the retired monolith decl.
  void emitIQ2XSCanonicalGridTableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit the DERIVED keven_signs_q2xs signs64 SIGN plane as ONE `static const int8_t
  /// tcrv_iq2xs_signs64[1024] = { ... };` verbatim decl -- the 128-entry ksigns_iq2xs
  /// selector (canonical kIQ2XSKsigns) EXPANDED to per-lane +-1 (byte b of selector j is
  /// `(ksigns[j] & (1<<b)) ? -1 : +1`). This IS the signs64 sign-plane mechanism carried by
  /// op identity (NO op-attr extension): the retired monolith derived it identically from
  /// its carried ksigns attr, so the decl is byte-identical.
  void emitIQ2XSCanonicalSigns64TableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit ONE iq2_xs super-block's per-half-scale GRID body at the current insertion point
  /// (INSIDE an already-open super-block loop whose per-super-block bases xb/yb are
  /// provided): the per-super-block `d = fp16(x.d)*y.d` fold scale, the qs/sc/q8 base setup,
  /// the `int32_t bsum = 0`, the flat 8-sub-block decode (the explicit scale byte ->
  /// ls1/ls2, the two-half 16-lane decode: the 2 grid indices + 2 sign selectors packed as
  /// u16 byte-offsets, the TWO vluxei16_v_i64m1 grid64/signs64 gathers + reinterpret i8m1 +
  /// the vmul_vv_i8m1 sign-onto-grid fold + vwmul_i16m2 + ONE vwredsum per half, then
  /// `bsum += sumi*ls`), and the per-super-block fp32 fold `sumf += d*(float)bsum` folded
  /// into the carried `sumf` lvalue. The trailing `*s = 0.125f*sumf` stays in the wrapper
  /// (OUT of this body). This is the byte-exact anchor kept across the iq2_xs flip: the
  /// retired monolith emitIQ2XSQ8KBlockDot's inline body was code-moved here (now the sole
  /// caller is the typed super-block SCALAR grid loop lowering) -- same nodes, same order.
  void emitIQ2XSSuperBlockGridBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const IQ2XSGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  /// The iq2_s super-block SCALAR-accumulator per-half-scale GRID loop emitter (the flip
  /// lowering, iq2_xs grid SIBLING, SIGN-PLANE explicit-signs variant, PER-HALF explicit
  /// scale): the iq2_s branch of the fold_model "scalar_delta_grid" path (dispatched by
  /// emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an iq2_s grid-core
  /// brick). It emits the wrapper -- the `static const int64_t tcrv_iq2s_grid[1024]` decl
  /// (from the canonical kIQ2SGrid), the UNIVERSAL `static const int8_t
  /// tcrv_iq2s_signs256[2048]` sign plane, the `sumf` float SCALAR accumulator seeded
  /// once OUTSIDE the loop, nb = n / QK_K, the ONCE (const int64_t *) grid64 + signs256
  /// views, the outer emitc.for over nb, the per-super-block base built from the brick's
  /// (base, block_index) via a shared memo (anti-bypass), and the trailing
  /// `*s = 0.125f*sumf` store -- delegating the in-loop per-super-block body to the shared
  /// emitIQ2SSuperBlockGridBody anchor (the fp16*fp32 d fold, the per-sub-block explicit
  /// 4-bit scale ls1/ls2, the qh-plane byte, the two-half 16-lane decode + the TWO
  /// vluxei16_v_i64m1 grid64/signs256 gathers + the vmul-onto-grid sign fold + signed
  /// widening dot + bsum fold, then sumf += d*(float)bsum). Like iq2_xs there is NO gearbox
  /// (fixed 16-lane per-half shape). Byte-identical to the retired monolith
  /// emitIQ2SQ8KBlockDot by construction (same decls, same body helper, same facts, same
  /// order) modulo the source-op provenance token + func name.
  mlir::LogicalResult emitTypedSuperBlockScalarDeltaGridLoopBodyIq2s(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      tcrvrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const;

  /// The bounded per-super-block byte-exact facts the iq2_s per-half-scale GRID body helper
  /// reads (the I4 mirror off the grid-core brick). The emitc element/pointer types are
  /// re-derived inside the helper from the MLIRContext (uniqued -> the SAME Type instances),
  /// so this struct carries only the provenance, the size type, the two buffer pointer
  /// types, the iq2_s format byte offsets/sub-block shape, and the two per-loop-invariant
  /// SSA views the sub-block gathers read: the `gridName` (const int64_t *) view of
  /// tcrv_iq2s_grid and the `signs256` (const int64_t *) view of tcrv_iq2s_signs256 (the
  /// UNIVERSAL explicit-sign-byte plane). Like iq2_xs there is NO coreLmul (fixed 16-lane
  /// per-half shape); iq2_s ADDS the signsOffset (the explicit sign-byte region at qs+32)
  /// and the qhOffset (the qh-bit plane) over iq2_xs.
  struct IQ2SGridBodyContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t weightDOffset;        //   0 (fp16 x.d)
    int64_t qsOffset;             //   2 (32 uint8 grid index bytes)
    int64_t signsOffset;          //  34 (32 uint8 EXPLICIT sign bytes = qs+QK_K/8)
    int64_t qhOffset;             //  66 (8 uint8 qh-bit plane bytes)
    int64_t scalesOffset;         //  74 (explicit 4-bit scales[8])
    int64_t activationDOffset;    //   0 (fp32 y.d)
    int64_t q8Offset;             //   4 (q8_K quants)
    int64_t subBlock;             //  32
    int64_t numSubBlocks;         //   8
    int64_t groupsPerSub;         //   4 (grid groups per sub-block, l=0..3)
    int64_t numGroupsPerHalf;     //   2 (grid/sign groups per 16-lane half)
    mlir::Value gridName;         // the (const int64_t *) view of tcrv_iq2s_grid
    mlir::Value signs256;         // the (const int64_t *) view of tcrv_iq2s_signs256
  };

  /// Emit the fixed 1024-entry iq2_s GRID codebook as ONE `static const int64_t
  /// tcrv_iq2s_grid[1024] = { ... };` verbatim decl (ggml's exact uint64 hex literals
  /// rendered `0x%016llxULL`), from the CANONICAL kIQ2SGrid constant (the grid-core brick
  /// carries no grid in the IR -- the emitter keys the fixed codebook off the brick op
  /// identity). Byte-identical to the retired monolith decl.
  void emitIQ2SCanonicalGridTableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit the UNIVERSAL signs256 SIGN plane as ONE `static const int8_t
  /// tcrv_iq2s_signs256[2048] = { ... };` verbatim decl -- every 8-bit sign byte value v
  /// EXPANDED to per-lane +-1 (byte b of value v is `(v & (1<<b)) ? -1 : +1`). iq2_s has
  /// NO ksigns selector: its signs are EXPLICIT bytes read straight from the sign region,
  /// so the gather is indexed by the raw 8-bit sign byte DIRECTLY (0..255). This IS the
  /// definitional bit-to-sign expansion the old scalar fold computed via
  /// vmv/vand/vmsne/vneg/vmerge; the retired monolith emitted the SAME universal table, so
  /// the decl is byte-identical (NO op-attr dependency -- the table is universal).
  void emitIQ2SCanonicalSigns256TableDecl(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  /// Emit ONE iq2_s super-block's per-half-scale GRID body at the current insertion point
  /// (INSIDE an already-open super-block loop whose per-super-block bases xb/yb are
  /// provided): the per-super-block `d = fp16(x.d)*y.d` fold scale, the qs/sgn/qh/sc/q8
  /// base setup, the `int32_t bsum = 0`, the flat 8-sub-block decode (the explicit scale
  /// byte -> ls1/ls2, the qh-plane byte, the two-half 16-lane decode: the 2 grid indices
  /// `qs[l] | ((qhb<<(8-2*l))&0x300)` + 2 explicit sign bytes packed as u16 byte-offsets,
  /// the TWO vluxei16_v_i64m1 grid64/signs256 gathers + reinterpret i8m1 + the
  /// vmul_vv_i8m1 sign-onto-grid fold + vwmul_i16m2 + ONE vwredsum per half, then
  /// `bsum += sumi*ls`), and the per-super-block fp32 fold `sumf += d*(float)bsum` folded
  /// into the carried `sumf` lvalue. The trailing `*s = 0.125f*sumf` stays in the wrapper
  /// (OUT of this body). This is the byte-exact anchor kept across the iq2_s flip: the
  /// retired monolith emitIQ2SQ8KBlockDot's inline body was code-moved here (now the sole
  /// caller is the typed super-block SCALAR grid loop lowering) -- same nodes, same order.
  void emitIQ2SSuperBlockGridBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const IQ2SGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  // NOTE: the monolith emitIQ1SQ8KBlockDot emitter was RETIRED at L3 iq1_s M3 (the
  // flip): the front door now constructs the typed super-block SCALAR-accumulator
  // GRID loop body (fold_model "scalar_delta_grid"), lowered by
  // emitTypedSuperBlockScalarDeltaGridLoopBody, which reuses the SHARED byte-exact
  // anchors below (emitIQ1SCanonicalGridTableDecl + emitIQ1SSuperBlockGridBody).

  /// The bounded per-super-block byte-exact facts the iq1_s TERNARY-grid body helper
  /// reads (the I4 mirror off the monolith op / the grid-core brick). The emitc
  /// element/pointer types are re-derived inside the helper from the MLIRContext
  /// (uniqued -> the SAME Type instances), so this struct carries only the
  /// provenance, the size type, the two buffer pointer types, the format byte
  /// offsets/sub-block shape, and the `tcrv_iq1s_grid` base literal.
  struct IQ1SGridBodyContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t weightDOffset;        //   0 (fp16 x.d)
    int64_t qsOffset;             //   2 (32 uint8 grid-index bytes)
    int64_t qhOffset;             //  34 (uint16 qh[8] plane)
    int64_t activationDOffset;    //   0 (fp32 y.d)
    int64_t q8Offset;             //   4 (q8_K quants)
    int64_t bsumsOffset;          // 260 (int16 per-sub-block sums)
    int64_t subBlock;             //  32
    int64_t numSubBlocks;         //   8
    int64_t groupsPerSub;         //   4 (grid groups per sub-block)
    mlir::Value gridArrayName;    // the `tcrv_iq1s_grid` u64 base literal
  };

  /// Emit the fixed 2048-entry iq1_s TERNARY grid codebook as ONE `static const
  /// uint64_t tcrv_iq1s_grid[2048] = { ... };` verbatim decl (ggml's exact hex
  /// literals rendered `0x%016llxULL`). The byte-exact SHARED anchor kept across the
  /// iq1_s flip: the retired monolith emitIQ1SQ8KBlockDot passed its carried grid attr;
  /// the typed grid loop lowering (the sole live caller) passes the canonical kIQ1SGrid
  /// (via emitIQ1SCanonicalGridTableDecl) so the emitted decl is byte-identical.
  void emitIQ1SGridTableDecl(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc,
                             llvm::ArrayRef<int64_t> grid) const;

  /// Emit the iq1_s grid decl from the CANONICAL kIQ1SGrid constant (the grid-core
  /// brick carries no grid in the IR -- the emitter keys the fixed codebook off the
  /// brick op identity). Delegates to emitIQ1SGridTableDecl; byte-identical to the
  /// monolith decl whose grid attr is populated from the same kIQ1SGrid.
  void emitIQ1SCanonicalGridTableDecl(mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Location loc) const;

  /// Emit ONE iq1_s super-block's TERNARY-grid body at the current insertion point
  /// (INSIDE an already-open super-block loop whose per-super-block bases xb/yb are
  /// provided): the per-super-block `d = fp16(x.d)*y.d` fold scale, the qs/qh/q8/bsums
  /// base setup, the two SCALAR i32 accumulators sumi + sumi1, the flat 8-sub-block
  /// grid gather (the 11-bit index build from qs+qh, the vluxei16 gather over the
  /// 2048-entry ternary grid, the signed widening product + ONE vwredsum per
  /// sub-block, the qh-scaled positive dot into sumi and the delta-bsum sum into
  /// sumi1), then the per-super-block scalar fold `sumf += d*((float)sumi +
  /// IQ1S_DELTA*(float)sumi1)` (IQ1S_DELTA=0.125f) folded into the carried `sumf`
  /// lvalue. This is the byte-exact anchor kept across the iq1_s flip: the retired
  /// monolith emitIQ1SQ8KBlockDot shared it with the typed super-block
  /// SCALAR-accumulator grid loop lowering (now the sole caller) -- same nodes, same
  /// order.
  void emitIQ1SSuperBlockGridBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const IQ1SGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  // NOTE: the monolith emitIQ1MQ8KBlockDot emitter was RETIRED at the iq1_m flip (L3):
  // the front door now constructs the typed super-block SCALAR-accumulator GRID loop
  // body (fold_model "scalar_delta_grid", stride 56), lowered by
  // emitTypedSuperBlockScalarDeltaGridLoopBodyIq1M, which reuses the SHARED byte-exact
  // anchors below (emitIQ1MCanonicalGridTableDecl + emitIQ1MSuperBlockGridBody).

  /// The bounded per-super-block byte-exact facts the iq1_m TERNARY-grid body helper
  /// reads (the I4 mirror off the monolith op / the iq1_m grid-core brick). The emitc
  /// element/pointer types are re-derived inside the helper from the MLIRContext
  /// (uniqued -> the SAME Type instances), so this struct carries only the
  /// provenance, the size type, the two buffer pointer types, the iq1_m format byte
  /// offsets/sub-block shape, and the `tcrv_iq1m_grid` base literal. iq1_m carries NO
  /// fp16 weight d (the scale is RECONSTRUCTED from the packed scales[] words) and NO
  /// bsums (the per-group delta reduces a fresh Σq8), so this struct has a
  /// weight_scales_byte_offset field (48) instead of iq1_s's bsums offset.
  struct IQ1MGridBodyContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t qsOffset;             //   0 (32 uint8 grid-index bytes, no fp16 d)
    int64_t qhOffset;             //  32 (16 uint8 qh bytes, 2 per sub-block)
    int64_t scalesOffset;         //  48 (4 uint16 packed scales[] words)
    int64_t activationDOffset;    //   0 (fp32 y.d)
    int64_t q8Offset;             //   4 (q8_K quants)
    int64_t subBlock;             //  32
    int64_t numSubBlocks;         //   8
    int64_t groupsPerSub;         //   4 (grid groups per sub-block)
    mlir::Value gridArrayName;    // the `tcrv_iq1m_grid` u64 base literal
  };

  /// Emit the fixed 2048-entry iq1_m TERNARY grid codebook as ONE `static const
  /// uint64_t tcrv_iq1m_grid[2048] = { ... };` verbatim decl (ggml's exact hex
  /// literals rendered `0x%016llxULL`). The SAME 2048 ternary iq1s_grid literals as
  /// iq1_s (just a distinct decl NAME so the two coexist). The byte-exact SHARED
  /// anchor: the retired monolith emitIQ1MQ8KBlockDot passed its carried grid attr;
  /// the typed grid loop lowering (the sole live caller) passes the canonical kIQ1MGrid
  /// (via emitIQ1MCanonicalGridTableDecl) so the emitted decl is byte-identical.
  void emitIQ1MGridTableDecl(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc,
                             llvm::ArrayRef<int64_t> grid) const;

  /// Emit the iq1_m grid decl from the CANONICAL kIQ1MGrid constant (the grid-core
  /// brick carries no grid in the IR -- the emitter keys the fixed codebook off the
  /// brick op identity). Delegates to emitIQ1MGridTableDecl; byte-identical to the
  /// monolith decl whose grid attr is populated from the same kIQ1MGrid.
  void emitIQ1MCanonicalGridTableDecl(mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Location loc) const;

  /// Emit ONE iq1_m super-block's TERNARY-grid body at the current insertion point
  /// (INSIDE an already-open super-block loop whose per-super-block bases xb/yb are
  /// provided): the qs/qh/sc/q8 base setup, the packed iq1m_scale fp16 RECONSTRUCT
  /// (`scbits = (sc[0]>>12)|((sc[1]>>8)&0xf0)|((sc[2]>>4)&0xf00)|(sc[3]&0xf000)` read
  /// as _Float16, `d = dx * y.d`), the two SCALAR i32 accumulators sumi1 + sumi2, the
  /// flat 8-sub-block grid dot (per-HALF vluxei16 gather over the 2048-entry ternary
  /// grid, signed widening product + ONE vwredsum per half into sum1[h]) with the TWO
  /// half scales ls1/ls2, the per-GROUP delta term (fresh Σq8 vwredsum x delta[l] into
  /// sum2[h], FOUR independent signs), then the per-super-block scalar fold `sumf +=
  /// d*((float)sumi1 + IQ1M_DELTA*(float)sumi2)` (IQ1M_DELTA=0.125f) folded into the
  /// carried `sumf` lvalue. The byte-exact anchor shared by the (retired) monolith
  /// emitIQ1MQ8KBlockDot and the typed super-block SCALAR-accumulator grid loop lowering
  /// (now the sole caller) -- same nodes, same order.
  void emitIQ1MSuperBlockGridBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const IQ1MGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_mxfp4_q8_0 block dot-product for one
  /// tcrv_rvv.mxfp4_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// raw()). It is the FP4 CODEBOOK sibling of emitIQ4NLQ8_0BlockDot, REUSING its
  /// codebook-gather integer core (vand 0x0F / vsrl 0x04 -> vrgather through the
  /// broadcast table -> the SAME emitOffsetBinaryProductFromDecodedValue product +
  /// vwredsum) verbatim. TWO kernel-specific facts differ:
  ///   (a) BLOCK FORMAT: block_mxfp4 = { uint8_t e; uint8_t qs[16] } (stride 17),
  ///       the FP4 nibbles at +1 (after the 1-byte E8M0 exponent); the q8_0
  ///       activation is unchanged (stride 34, quants at +2, high half at +16).
  ///       So the weight quant offset (1) and activation quant offset (2) are
  ///       DISTINCT (iq4_nl shared +2).
  ///   (b) WEIGHT SCALE: NOT a fp16 read. The structured E8M0 -> fp32 HALF
  ///       reconstruction GGML_E8M0_TO_FP32_HALF(e) = 2^(e-128): a uint32_t
  ///       `bits = (e < 2) ? (0x00200000u << e) : ((e - 1) << 23)` (the e<2 branch
  ///       the denormal scales 2^-128/2^-127, the e>=2 branch the normalized
  ///       2^(e-128)) reinterpreted as a float via `*(const float *)&bits` -- ggml's
  ///       EXACT bit construction (no scalbnf/ldexpf, byte-identical on denormals).
  ///       Only ONE fp16 read survives (the q8_0 d_y).
  ///
  ///   static const int8_t tcrv_mxfp4_kvalues[16] = { ... };  // FP4 codebook decl
  ///   float sumf = 0.0f;  size_t nb = n / 32;
  ///   vint8m1_t values = __riscv_vle8_v_i8m1(tcrv_mxfp4_kvalues, 16);   // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*17;  const uint8_t *yb = vy + ib*34;
  ///     uint8_t  e   = *(const uint8_t *)(xb);                // E8M0 exponent
  ///     uint32_t bits = (e < 2) ? (0x00200000u << e) : ((e - 1) << 23);
  ///     float scale_x = *(const float *)&bits;               // 2^(e-128)
  ///     float d_y = (float)*(const _Float16 *)(yb);          // q8_0 fp16 scale
  ///     int32_t sumi = 0;  size_t vl = __riscv_vsetvl_e8m1(16);
  ///     vuint8m1_t w  = __riscv_vle8_v_u8m1(xb + 1, vl);      // FP4 nibbles @ +1
  ///     vint8m1_t  y0 = __riscv_vle8_v_i8m1(yb + 2, vl);      // q8[0..15] @ +2
  ///     vint8m1_t  y1 = __riscv_vle8_v_i8m1(yb + 18, vl);     // q8[16..31] @ +18
  ///     ... codebook gather + vwmul/vwmacc + vwredsum -> sumi ...
  ///     sumf = sumf + (float)sumi * (scale_x * d_y);    // ggml mxfp4 order
  ///   }
  ///   *s = sumf;
  /// The codebook GATHER pins the m1 anchor (VLMAX >= 16 to index all 16 entries);
  /// the verifier enforces it (the FP4 codebook class is inherently Zvl128b-gated).
  mlir::LogicalResult emitMXFP4Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_nvfp4_q8_0 block dot-product for one
  /// tcrv_rvv.nvfp4_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// raw()). NVFP4 (NVIDIA FP4) is the SECOND FP4-class sibling: it REUSES mxfp4's
  /// FP4 codebook gather + the asymmetric signed-widening integer core verbatim.
  /// The THREE genuinely-new facts:
  ///   (a) SUPER-BLOCK FORMAT: block_nvfp4 = { uint8_t d[4]; uint8_t qs[32] }
  ///       (QK=64, stride 36): four UE4M3 sub-block scales at +0..3, then 32 FP4
  ///       nibble bytes at +4 (8 per sub-block, sub-block s at +4 + s*8). One
  ///       super-block is four 16-element sub-blocks spanning TWO block_q8_0 blocks.
  ///   (b) PER-SUB-BLOCK UE4M3 -> fp32 scale (the new piece): UE4M3 is unsigned
  ///       (4 exp bias-7 / 3 man). e==0||e==0x7F -> 0.0f; else exp=(e>>3)&0xF,
  ///       man=e&7, raw = (exp==0) ? ldexpf(man,-9) : ldexpf(1+man/8, exp-7); * 0.5f
  ///       (the HALF compensation for the doubled kvalues_mxfp4 codebook). Emitted
  ///       as structured emitc (no E8M0 bit dance -- UE4M3 scales are all normal
  ///       fp32, so ggml's exact ldexpf arithmetic is byte-faithful by construction).
  ///   (c) ACTIVATION ADDRESSING: sub-block s reads q8 block (2*ib + s/2) at the
  ///       half-offset (s%2)*16 within that q8_0's 32 quants; the strip is 8 lanes
  ///       (qk_sub/2), the high nibble at +8.
  ///
  /// _generic shape (quants.c:278-312):
  ///   static const int8_t kvalues_mxfp4[16] = { ... };
  ///   vint8m1_t values = __riscv_vle8_v_i8m1(kvalues_mxfp4, 16);   // ONCE
  ///   float sumf = 0;
  ///   for (ib = 0; ib < nb; ++ib) {                                // super-blocks
  ///     for (s = 0; s < 4; ++s) {                                  // sub-blocks
  ///       const uint8_t *xb = vx + ib*36;
  ///       float d  = ggml_ue4m3_to_fp32(xb[s]);                    // (b)
  ///       const uint8_t *xqs = xb + 4 + s*8;                       // FP4 nibbles
  ///       const uint8_t *yb = vy + (2*ib + s/2)*34;                // (c)
  ///       float dy = (float)*(const _Float16 *)yb;
  ///       const int8_t *yqs = (const int8_t *)(yb + 2 + (s%2)*16);
  ///       // strip 8 lanes: gather xqs (lo<->yqs[0..7], hi<->yqs[8..15]) -> sumi
  ///       sumf = sumf + dy * d * (float)sumi;                      // ggml order
  ///     }
  ///   }
  ///   *s = sumf;
  mlir::LogicalResult emitNVFP4Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// The bounded set of structural facts + EmitC types the q6_K super-block
  /// integer core needs. Shared by the K1 (aux32 partial) and K2 (full block
  /// dot) emitters so both drive the SAME unpack + sub-block-loop nodes; the
  /// integer core is byte-pinned by the K1 ssh-rvv artifact and must stay
  /// node-identical for either caller.
  struct Q6_KIntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type i32ImmType;
    mlir::Type i8ElemType;
    mlir::Type u8m2Type;
    mlir::Type i8m2Type;
    mlir::Type i8mf2Type;
    mlir::Type i16m1Type;
    mlir::Type i32m2Type;
    mlir::Type i8PtrType;
    mlir::Type u8PtrType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t subBlock;
    int64_t qhOffset;
    int64_t scalesOffset;
    int64_t q8Offset;
    int64_t numSubBlocks;
    int64_t half;
    // The optional integer_core_lmul knob (default "mf2" == today's emit:
    // TWO 8-lane halves per 16-element sub-block). l8/l16/l32 name the Region-B
    // MAC chain widths: l8 = the per-strip i8 load anchor, l16 = the vwmul i16
    // product, l32 = the vwmacc i32 accumulator. mf2: {mf2,m1,m2} (two 8-lane
    // halves). m1: {m1,m2,m4} (ONE 16-lane strip per sub-block, folded back to
    // 8). stripWidth = the per-sub-block MAC strip element count (8 @mf2, 16
    // @m1); foldGroups = stripWidth/8 (1 @mf2 -> no fold; 2 @m1 -> fold the wide
    // aux32 back to 8 element-wise BEFORE the fp32 cvt, VLEN-agnostically).
    llvm::StringRef coreLmul;
    llvm::StringRef l8;
    llvm::StringRef l16;
    llvm::StringRef l32;
    mlir::Type i8WideType;   // vint8<l8>   (the strip load type)
    mlir::Type i16WideType;  // vint16<l16> (the vwmul product type)
    mlir::Type i32WideType;  // vint32<l32> (the wide aux32 accumulator type)
    int64_t stripWidth;      // 8 @mf2, 16 @m1
    int64_t foldGroups;      // stripWidth / 8: 1 @mf2 (no fold), 2 @m1
  };

  /// Emit ONE super-block's integer core (the 6-bit ql+qh unpack into the
  /// element-ordered aux8[256] scratch, biased -32, then the nested sub-block
  /// loop applying the per-sub-block int8 scale in the i32 domain) at the
  /// current insertion point (INSIDE an already-open super-block loop body),
  /// and RETURN the per-super-block aux32[8] integer-state vector (vint32m2).
  /// This is the genuinely-new super-block integer machinery, shared verbatim
  /// by K1 (which then vse32-stores the returned vector) and K2 (which folds it
  /// into the deferred fp32 accumulator). The emission mirrors _generic
  /// (quants.c:826-847) node-for-node: the aux32 accumulator is RESET to zero
  /// each super-block, the unpack carries the entire ql/qh->element permutation
  /// (the dot then reads aux8 contiguously), and the 8-lane accumulation is
  /// integer (order-free). RETURNS the aux32 lvalue VARIABLE (not a load), so
  /// each caller can materialize the load at its own statement position (K1's
  /// emitted C is byte-pinned by its ssh-rvv artifact).
  mlir::TypedValue<emitc::LValueType> emitQ6_KSuperBlockAux32Core(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q6_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base) const;

  /// Shared context for the q3_K super-block integer core (the q3_K sibling of
  /// Q6_KIntegerCoreContext). It carries the block-format FACTS + the ABI pointer
  /// types the 2-bit/subtractive-hmask unpack + the signed 6-bit scale dance + the
  /// per-sub-block i32 dot read; the EmitC vector/pointer TYPES and the widening
  /// chain (deriveWideningChain(coreLmul)) are derived INSIDE the core helper (from
  /// the MLIRContext + coreLmul), so the caller supplies only the mf2/m1 knob.
  struct Q3_KIntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t subBlock;
    int64_t hmaskOffset;
    int64_t qsOffset;
    int64_t scalesOffset;
    int64_t q8Offset;
    int64_t numSubBlocks;
    int64_t half;
    // The optional integer_core_lmul knob (default "mf2" == today's emit: TWO
    // 8-lane halves per 16-element sub-block). "m1" runs ONE 16-lane strip per
    // sub-block, folded back to the canonical 8 BEFORE returning (VLEN-agnostic
    // vslidedown by literal element offset 8 + vadd + vget(.,0)). The subtractive
    // hmask unpack + the signed 6-bit scale dance are LMUL-free at every knob.
    llvm::StringRef coreLmul;
  };

  /// Emit ONE super-block's q3_K integer core (the 2-bit + SUBTRACTIVE-hmask unpack
  /// into the element-ordered aux8[256] scratch, SIGNED [-4,3]; the q3_K-OWN SIGNED
  /// 6-bit scale bit-dance staged through the caller-supplied uint32_t utmp[4]
  /// scratch; then the nested sub-block loop applying the per-sub-block signed
  /// `scales[js]-32` scale in the i32 domain) at the current insertion point
  /// (INSIDE an already-open super-block loop body), and RETURN the per-super-block
  /// aux32[8] integer-state vector (canonical 8-lane vint32m2 lvalue). Extracted
  /// VERBATIM from the retired emitQ3_KQ8_KBlockDot monolith so the emitted C is
  /// byte-identical by construction. The deferred no-min fp32 fold is DELIBERATELY
  /// out of scope (it is the reused q6_K positive fold the single-vector lowering
  /// applies). Mirrors emitQ6_KSuperBlockAux32Core (different unpack -- hmask
  /// predicated decrement, not q6_K's ql+qh -- and a SIGNED 6-bit scale, not q6_K's
  /// direct int8), so at mf2 (the only knob the aux32 op stamps) it returns the
  /// 8-lane aux32Var directly (no fold-back tokens), byte-identical to the monolith.
  mlir::TypedValue<emitc::LValueType> emitQ3_KSuperBlockAux32Core(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q3_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base,
      mlir::TypedValue<emitc::ArrayType> utmpArray) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_q1_0_q8_0 block dot-product for one
  /// tcrv_rvv.q1_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-control-flow blob -- every value is a node in the IR graph). It
  /// is the BINARY ({-1,+1}) class: each q1_0 weight bit is a SIGN (set -> +q8,
  /// clear -> -q8) and the q8 value itself is the magnitude. It runs ggml's
  /// shipped _vl128 lane structure -- ONE 32-lane sub-block body with NO kmask
  /// table, NO vand/vmsne decode, NO codebook, NO nibble unpack, NO offset-binary
  /// `-8` bias. The structured shape is:
  ///   float sumf = 0.0f;  size_t nb = n / 128;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;
  ///     float d0 = (float)*(const _Float16 *)(xb);
  ///     float sumi = 0.0f;
  ///     for (k = 0; k < 4; ++k) {                      // UNROLLED sub-blocks
  ///       const uint8_t *yb = vy + (ib*4 + k)*34;
  ///       float d1 = (float)*(const _Float16 *)(yb);
  ///       size_t vl = __riscv_vsetvl_e8<anchor>(32);   // anchor: m2@128 / m1@256
  ///       // the 4 packed bit-bytes ARE the i8 sign mask (bit 8b+i -> lane 8b+i):
  ///       vbool<r>_t m = __riscv_vlm_v_b<r>(xb + 2 + k*4, vl); // r=4 (m2)/8 (m1)
  ///       vint8<a>_t q8 = __riscv_vle8_v_i8<a>(yb + 2, vl);
  ///       vint8<a>_t sy = __riscv_vmerge_vvm_i8<a>(            // i8-DOMAIN sign
  ///           __riscv_vneg_v_i8<a>(q8, vl), q8, m, vl);       // +q8 if bit set
  ///       int sumi_block = __riscv_vmv_x_s_i16m1_i16(          // ONE reduce/block
  ///           __riscv_vwredsum_vs_i8<a>_i16m1(sy, vmv_v_x_i16m1(0,1), vl));
  ///       sumi = sumi + d1 * (float)sumi_block;          // ggml exact order
  ///     }
  ///     sumf = sumf + d0 * sumi;                          // ggml exact order
  ///   }
  ///   *s = sumf;
  /// The negate/merge is in the i8 DOMAIN (ggml _vl128 exact); the real q8 quant
  /// domain is [-127,127] so the i8 vneg is byte-exact on every gate input (only
  /// the unreachable -128 boundary would overflow -- matching ggml _vl128, not the
  /// old i16-widen superset). The anchor MOVES with VLEN: the 32-element sub-block
  /// straddles m1's i8 VLMAX boundary between 128/256, so the gearbox stamps m2 at
  /// VLEN128 (e8m1 VLMAX 16 < 32) and the lighter m1 at VLEN256; the default is the
  /// VLEN-universal-safe m2. The 32-lane reduce is pure within-block lowering (vl
  /// stays 32, never crosses into a second sub-block). The block-format facts are
  /// the op's typed attrs (I4 mirror); the emission is the op's fixed structure.
  /// The five fp16 reads (d0 + d1_{0..3}) are the only sanctioned opaque scalar
  /// pieces.
  mlir::LogicalResult emitQ1_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the ggml ggml_vec_dot_q6_K_q8_K INTEGER CORE (the K-quant K1
  /// increment) for one tcrv_rvv.q6_k_q8_k_aux32_partial op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob). It reproduces the
  /// per-super-block aux32[8] integer state EXACTLY as _generic computes it right
  /// before the fp32 d-multiply, byte-exact, and stores it through the output
  /// pointer (NO fp32 fold -- that is K2):
  ///   size_t nb = n / 256;
  ///   int8_t aux8[256];
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*210;  const uint8_t *yb = vy + ib*292;
  ///     // (A) unpack 256 weights into aux8 (element-ordered, biased -32):
  ///     //     two 128-element chunks, each a 4-strip 32-wide e8m2 vand/vsrl/
  ///     //     vsll/vor/vsub reconstruct + vse8 store at the matching offset.
  ///     // (B) per-sub-block int8-scaled i32 dot into the 8-lane aux32 vector:
  ///     vint32m2_t aux32 = vmv_v_x_i32m2(0, 8);   // RESET per super-block
  ///     for (size_t js = 0; js < 16; js += 1) {
  ///       int32_t scale = scales[js];             // int8 -> i32 scalar
  ///       // two halves of 8: vwmul i8xi8->i16 then vwmacc.vx i32 += scale*i16
  ///     }
  ///     vse32_v_i32m2(out + ib*8, aux32, 8);      // store aux32[8] state
  ///   }
  /// The unpack carries the ENTIRE ql/qh->element permutation (the dot then reads
  /// aux8 contiguously exactly as _generic's `a += 8`); the per-sub-block scale is
  /// applied in the i32 domain (vwmacc.vx); the 8-lane aux32 accumulation is
  /// order-free (integer add). The block-format facts are the op's typed attrs
  /// (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ6_KQ8_KAux32Partial(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Shared context for the q4_K super-block integer core (K4a + K4b), carrying
  /// the EmitC types + the block-format facts the unpack / bit-dance / sub-block
  /// dot need. The factoring mirrors q6_K's Q6_KIntegerCoreContext so the K4a
  /// emitted C stays byte-identical when K4b reuses the same core.
  struct Q4_KIntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type i32ImmType;
    mlir::Type u32Type;
    mlir::Type i8ElemType;
    mlir::Type u8m2Type;
    mlir::Type i8m2Type;
    mlir::Type i8mf2Type;
    mlir::Type i16m1Type;
    mlir::Type i32m2Type;
    mlir::Type i8PtrType;
    mlir::Type u8PtrType;
    mlir::Type constU32Type;
    mlir::Type u32PtrType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t subBlock;
    int64_t scalesOffset;
    int64_t qsOffset;
    int64_t q8Offset;
    int64_t numSubBlocks;
    int64_t quarter;
    // The integer-core MAC chain (Region C) LMUL anchor, sourced from the op's
    // optional integer_core_lmul (default "mf2" == today's emit). l8/l16/l32 are
    // the three element-width rungs of the i8 -> i16 -> i32 widening chain. The
    // default (mf2 -> m1 -> m2) reproduces the byte-identical legacy form; m1 ->
    // (m1, m2, m4); m2 -> (m2, m4, m8). The Region-A 6-bit unpack (u8m2/i8m2) is
    // NOT driven by this knob -- it stays fixed at m2 (its own throughput axis,
    // widened in a later increment), so the Region-A types/callees are hardcoded.
    llvm::StringRef coreLmul = "mf2";
    llvm::StringRef l8 = "mf2";
    llvm::StringRef l16 = "m1";
    llvm::StringRef l32 = "m2";
    // S3-S4 Region-C widen: the per-sub-block MAC strip width derived from l8
    // (mf2 -> 8, m1 -> 16, m2 -> 32) and the resulting strip count over the
    // 32-element sub-block (32/stripWidth = 4/2/1). foldGroups = stripWidth/8 is
    // the number of canonical-8-lane groups the wide aux32 must be folded back
    // into (1 at mf2 == no fold). i32Canon8Type is the CANONICAL 8-lane
    // vint32m2_t the fold-back collapses into -- it is the byte-exact fp-fold
    // (Region F) contract type and stays vint32m2_t at EVERY l8 (the wide aux32
    // running type is cx.i32m2Type = vint32<l32>). At mf2 stripWidth==8,
    // foldGroups==1, i32Canon8Type==i32m2Type, so the wide path is unreached and
    // the legacy emitQuarter path runs byte-identically.
    int64_t stripWidth = 8;
    int64_t numStrips = 4;
    int64_t foldGroups = 1;
    mlir::Type i32Canon8Type;
    // The q5_K 5th-bit (qh high-bit-plane) injection, the ONLY q5_K vs q4_K
    // difference. When hasQh is false (q4_K), the core emits BYTE-IDENTICAL
    // nodes (no qh load, no inject). When true (q5_K), each unpacked nibble gets
    // `+16` added in the UINT8 domain (before the u8->i8 reinterpret) for every
    // element whose qh bit is set: per 32-element half h in 0..7, the bit for
    // element l is `(qh[l] >> h) & 1`. The qh plane is loaded ONCE per
    // super-block from xb + qhOffset and reused across all 8 halves.
    bool hasQh = false;
    int64_t qhOffset = 0;
  };

  /// The q4_K super-block integer core's result handles: the per-super-block
  /// aux32[8] integer-state lvalue (vint32m2, RESET each super-block) AND the
  /// const uint8_t* into the decoded utmp -- the 8 6-bit scales (bytes 0..7)
  /// followed by the 8 6-bit mins (bytes 8..15), the same type-pun _generic
  /// does with scales=&utmp[0]/mins=&utmp[2]. K4a vse8-stores both scales+mins
  /// and vse32-stores aux32; K4b folds aux32 into the deferred fp32 sums and
  /// reads the mins for the min term -- so the helper exposes both.
  struct Q4_KCoreResult {
    mlir::TypedValue<emitc::LValueType> aux32Var;
    mlir::Value scalesU8;
  };

  /// The Track B q4_K BRICK 4 (MIN-term) context: the bounded facts + emitc
  /// types the two MIN-term sub-region helpers (emitQ4_KMinTermBsumsDot +
  /// emitQ4_KMinTermSubtract) need. The MIN term is a SCALAR integer reduction
  /// (sumi = sum_j(bsums[j] * mins[j/2])) plus a single fp contraction
  /// (sumf -= dmin * (float)sumi) -- no LMUL/widening axis, so this is a small
  /// dedicated context (NOT the integer-core context, whose fields are all about
  /// the vector MAC chain). Carrying the (deterministically-interned) emitc types
  /// by value keeps the monolithic q4_K block-dot's existing type locals in use
  /// (zero removals) and guarantees the helper references the SAME Type instances
  /// the monolith does -- byte-identity by construction.
  struct Q4_KMinTermContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type floatType;
    mlir::Type i32Type;       // emitc.opaque<"int"> (the sumi scalar type)
    mlir::Type constU8Type;   // emitc.opaque<"const uint8_t"> (the mins load)
    mlir::Type constI16Type;  // emitc.opaque<"const int16_t"> (the bsums load)
    mlir::Type constI16PtrType;
    llvm::StringRef fp16ReadCallee; // "(float)*(const _Float16 *)"
    int64_t bsumsOffset;            // 260 (q8_K block bsums byte offset)
    int64_t weightDminOffset;       //   2 (block_q4_K/q5_K dmin fp16 offset)
    int64_t numBsums;               //  16 (QK_K / 16)
  };

  /// The Track B q4_K BRICK 6 (positive-fold) context: the bounded facts + emitc
  /// types the shared positive-fold helper (emitQ4_KSumsFoldScaleD) needs. The
  /// fold is a fixed-order 8-lane fp32 sequence (vfcvt the canonical-8 aux32,
  /// vfmul.vf by the per-super-block scale d, vfadd.vv into the carried sums) --
  /// no LMUL/widening axis (BRICK 3 already folded aux32 back to canonical-8), so
  /// this is a small dedicated context. Carrying the (deterministically-interned)
  /// emitc types by value keeps the monolithic q4_K block-dot's existing type
  /// locals in use (zero removals) and guarantees the helper references the SAME
  /// Type instances the monolith does -- byte-identity by construction.
  struct Q4_KSumsFoldContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type floatType;     // emitc.opaque<"float"> (the d product scalar)
    mlir::Type f32m2Type;     // vfloat32m2_t (the 8-lane sums accumulator)
    mlir::Type i32Canon8Type; // vint32m2_t (the BRICK 3 canonical-8 aux32)
    llvm::StringRef fp16ReadCallee; // "(float)*(const _Float16 *)"
    int64_t weightDOffset;          // 0 (block_q4_K/q5_K d fp16 offset)
  };

  /// The Track B q4_K BRICK 7 (post-loop horizontal fold) context: the bounded
  /// facts + emitc types the shared horizontal-fold helper (emitQ4_KHorizontalFold)
  /// needs. The fold is a fixed-order 8-lane collapse (vse32 the carried 8-lane
  /// fp32 sums into a sums8[8] scratch, then the SEQUENTIAL ascending `sumf +=
  /// sums8[l]` for l = 0..7) -- no LMUL/widening axis (the horizontal collapse is
  /// fixed at 8 lanes f32m2), so this is a small dedicated context. Carrying the
  /// (deterministically-interned) emitc types by value keeps the monolithic q4_K
  /// block-dot's existing type locals in use (zero removals) and guarantees the
  /// helper references the SAME Type instances the monolith does -- byte-identity
  /// by construction.
  struct Q4_KHorizontalFoldContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type floatType; // emitc.opaque<"float"> (sumf + sums8 lanes)
    mlir::Type f32m2Type; // vfloat32m2_t (the carried 8-lane sums accumulator)
    int64_t numLanes;     // 8 (the canonical fp32 sums lane count)
  };

  /// Emit ONE super-block's q4_K/q5_K plain 4-bit nibble unpack into the
  /// element-ordered aux8[256] scratch (Region A), NO bias: FOUR 64-element
  /// chunks, each one vsetvl_e8m2(32) + vle8_v_u8m2 of the 32 weight bytes at
  /// `xb + qsOffset + chunk*32`, a 32-wide vand 0x0F (low nibble -> a[chunk*64 +
  /// 0..31]) and a 32-wide vsrl 4 (high nibble -> a[chunk*64 + 32..63]), each
  /// reinterpreted u8m2 -> i8m2 and vse8-stored into aux8Array via emitc
  /// SubscriptOp. q5_K (cx.hasQh) additionally injects the qh 5th bit in the u8
  /// domain before the reinterpret. This is the Track B q4_K BRICK 1 vocabulary:
  /// the SAME node sequence is shared VERBATIM by emitQ4_KSuperBlockAux32Core
  /// (the monolithic q4_K/q5_K integer core, which calls this in-loop after the
  /// per-super-block address arithmetic) AND the first-class
  /// tcrv_rvv.q4_k_nibble_unpack op's lowering (which declares its own aux8 and
  /// decodes one super-block), so both emit byte-identical Region-A C. Writes
  /// aux8Array as a side effect; returns nothing.
  void emitQ4_KPlainNibbleUnpack(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KIntegerCoreContext &cx, mlir::Value xb,
      mlir::TypedValue<emitc::ArrayType> aux8Array) const;

  /// Emit ONE super-block's q4_K/q5_K 6-bit scale/min bit-dance (Region B) into
  /// the uint32_t utmp[4] scratch: three (const uint32_t *)(xb + scalesOffset)
  /// word loads w0/w1/w2, the kmask1/2/3 (0x3f3f3f3f / 0x0f0f0f0f / 0x03030303)
  /// cross-byte shuffle via STRUCTURED scalar emitc.bitwise_{and,or,left_shift,
  /// right_shift} into utmp[0..3] (the 16 bytes [scales[0..7], mins[0..7]]),
  /// mirroring _generic (quants.c:685-690). Returns scalesU8 = (const uint8_t
  /// *)&utmp[0]. This is the Track B q4_K BRICK 2 vocabulary: the SAME node
  /// sequence is shared VERBATIM by emitQ4_KSuperBlockAux32Core (the monolithic
  /// q4_K/q5_K integer core, which calls this in-loop after the Region-A unpack)
  /// AND the first-class tcrv_rvv.q4_k_scale_min_bit_dance op's lowering (which
  /// declares its own utmp and decodes one super-block), so both emit
  /// byte-identical Region-B C. Writes utmpArray as a side effect.
  mlir::Value emitQ4_KScaleMinBitDanceCore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KIntegerCoreContext &cx, mlir::Value xb,
      mlir::TypedValue<emitc::ArrayType> utmpArray) const;

  /// Emit ONE super-block's q4_K/q5_K per-sub-block uint6-scaled i32 dot into the
  /// 8-lane aux32 PLUS the C-tail integer fold-back (Region C): DECLARE the
  /// per-super-block aux32 lvalue, seed it to zero via vmv_v_x_i32<l32>, run the
  /// sub-block loop applying the per-sub-block UINT6 scale `scalesU8[js]` in the
  /// i32 domain through the integer_core_lmul MAC strip (vsetvl_e8<l8> / two vle8
  /// (q8 + aux8) / vwmul i16<l16> / vwmacc i32<l32>, cx.numStrips per 32-element
  /// sub-block), then -- when cx.foldGroups > 1 (the wide m1/m2 anchors) -- the
  /// VLEN-agnostic integer fold-back of the WIDE aux32's group-of-8 residues to
  /// the canonical 8-lane vint32m2_t (vslidedown + vadd register-only regroup +
  /// vget). The scale is FUSED into the vwmacc; this whole MAC+fold region reads
  /// scalesU8 (the BRICK 2 decoded scales pointer) as ONE unit, the aux8 base
  /// (the BRICK 1 unpack scratch) and the q8 activation base derived from `yb`
  /// via cx.q8Offset / cx.activationPtrType. Integer add is associative/order-free
  /// so the regroup is provably bit-exact at every legal LMUL (NO fp
  /// non-associativity -- that is Region F, deferred). Returns the canonical-8
  /// aux32 lvalue (the byte-exact Region-F contract type). This is the Track B
  /// q4_K BRICK 3 vocabulary: the SAME node sequence is shared VERBATIM by
  /// emitQ4_KSuperBlockAux32Core (the monolithic q4_K/q5_K integer core, which
  /// calls this in-loop after the Region-A unpack + Region-B bit-dance) AND the
  /// first-class tcrv_rvv.q4_k_scaled_dot op's lowering (which passes the aux8 /
  /// scales / q8 ABI input pointers and stores the canonical-8 aux32 observable),
  /// so both emit byte-identical Region-C C.
  mlir::TypedValue<emitc::LValueType> emitQ4_KScaledDotIntoAux32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KIntegerCoreContext &cx, mlir::Value yb, mlir::Value aux8Base,
      mlir::Value scalesU8) const;

  /// Emit the FIRST half of ONE super-block's q4_K/q5_K MIN term (Track B BRICK
  /// 4): the int16 bsums load (yb + bsumsOffset, cast const int16_t *) and the
  /// SCALAR integer reduction `int sumi = 0; for (j) sumi += bsums[j] *
  /// mins[j/2]` where the decoded uint6 mins are scalesU8[8 + j/2] (mins =
  /// scalesU8 + 8; each min spans TWO consecutive bsums). Emits at the current
  /// insertion point and RETURNS the per-super-block scalar sumi lvalue. The
  /// integer reduction is associative/order-free (byte-exact at any tree). This
  /// is the Track B q4_K BRICK 4 vocabulary half: the SAME node sequence is
  /// shared VERBATIM by emitQ4_KQ8_KBlockDot (the monolithic q4_K/q5_K block dot,
  /// which emits it BEFORE its deferred positive fold) AND the first-class
  /// tcrv_rvv.q4_k_min_term op's lowering, so both emit byte-identical C. (The
  /// MIN term's two halves are interleaved with the positive fold in the monolith
  /// loop -- the reduction here, the subtract after -- but are data-independent,
  /// so the carve is two byte-exact helpers, NOT one contiguous span.)
  mlir::TypedValue<emitc::LValueType> emitQ4_KMinTermBsumsDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KMinTermContext &mx, mlir::Value yb, mlir::Value scalesU8) const;

  /// Emit the SECOND half of ONE super-block's q4_K/q5_K MIN term (Track B BRICK
  /// 4): the fp16 dmin read (`dmin = fp16(*(const _Float16 *)(xb +
  /// weightDminOffset)) * dy`) and the single fp contraction `sumf = sumf - dmin
  /// * (float)sumi` rendered as ONE emitc.expression (ggml's `sumf -= dmin *
  /// sumi`, quants.c:714, the emitc.load temps staying OUTSIDE the expression),
  /// then the AssignOp into the carried sumf lvalue. `dy` is the once-loaded fp32
  /// activation scale (the monolith loads it at fold_activation_d, shared with
  /// the positive fold; the standalone op loads it in its wrapper). The final
  /// `dmin * sumi` is a single fixed-order scalar fp multiply -- no
  /// fp-reassociation seam. The SAME node sequence is shared VERBATIM by
  /// emitQ4_KQ8_KBlockDot (which emits it AFTER its positive fold) AND the
  /// first-class tcrv_rvv.q4_k_min_term op's lowering, so both emit
  /// byte-identical C.
  void emitQ4_KMinTermSubtract(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KMinTermContext &mx, mlir::Value xb, mlir::Value dy,
      mlir::TypedValue<emitc::LValueType> sumiVar,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  /// Emit ONE super-block's q4_K/q5_K DEFERRED fp32 POSITIVE fold (Track B BRICK
  /// 6): `float d = fp16(*(const _Float16 *)(xb + weightDOffset)) * dy` (the fp16
  /// weight super-block scale times the once-loaded fp32 activation scale), then
  /// the fixed-order 8-lane sequence `af = vfcvt_f_x_v_f32m2(aux32, 8)` /
  /// `pr = vfmul_vf_f32m2(af, d, 8)` (a SEPARATE multiply, NEVER an fma) /
  /// `sums = vfadd_vv_f32m2(sums, pr, 8)` (a SEPARATE add) and the AssignOp into
  /// the carried sums lvalue. Reads the BRICK 3 canonical-8 aux32 lvalue (always
  /// vint32m2_t) -- the per-sub-block uint6 scale is NOT read here (BRICK 3 fused
  /// it into aux32), so there is NO fp-reassociation seam in this brick. Emits at
  /// the current insertion point. The SAME node sequence is shared VERBATIM by
  /// emitQ4_KQ8_KBlockDot (the monolithic q4_K/q5_K block dot, which emits it
  /// in-loop BETWEEN the two interleaved-but-data-independent MIN-term halves) AND
  /// the first-class tcrv_rvv.q4_k_sums_fold_scale_d op's lowering, so both emit
  /// byte-identical positive-fold C. `dy` is the once-loaded fp32 activation scale
  /// (the monolith loads it at fold_activation_d, shared with the MIN subtract;
  /// the standalone op loads it in its wrapper). The weight pointer type is
  /// derived from xb.
  void emitQ4_KSumsFoldScaleD(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KSumsFoldContext &sx, mlir::Value xb, mlir::Value dy,
      mlir::TypedValue<emitc::LValueType> aux32Var,
      mlir::TypedValue<emitc::LValueType> sumsVar) const;

  /// Emit ONE super-block-loop's POST-LOOP horizontal fold (Track B q4_K BRICK 7):
  /// `vse32_v_f32m2(&sums8[0], sums, 8)` materializes the carried 8-lane fp32 sums
  /// accumulator into the sums8[8] scratch, then the FIXED sequential ascending
  /// `sumf = sumf + sums8[l]` for l = 0..num_lanes-1 (num_lanes == 8) collapses the
  /// 8 lanes into the carried scalar `sumf` (which already holds the in-loop MIN
  /// subtractions). The horizontal sum is anchor-INDEPENDENT (always 8 lanes, fixed
  /// ascending order, NEVER a vfredusum) -- it mirrors `_generic`'s
  /// `for (l = 0; l < 8; ++l) sumf += sums[l]`, so there is NO fp-reassociation seam
  /// in this brick. Emits at the current insertion point (AFTER the super-block
  /// loop). Reads the carried sums + sumf lvalues and the sums8 scratch; does NOT
  /// emit the final store (the monolith stores `*s = sumf` through the ABI float *,
  /// the standalone op stores into a local sink) -- it RETURNS the final sumf Value
  /// so each caller can store it. The SAME node sequence is shared VERBATIM by
  /// emitQ4_KQ8_KBlockDot (the monolithic q4_K block dot, which emits it after its
  /// super-block loop) AND the first-class tcrv_rvv.q4_k_horizontal_fold op's
  /// lowering, so both emit byte-identical horizontal-fold C.
  mlir::Value emitQ4_KHorizontalFold(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KHorizontalFoldContext &hx,
      mlir::TypedValue<emitc::ArrayType> sums8Array,
      mlir::TypedValue<emitc::LValueType> sumsVar,
      mlir::TypedValue<emitc::LValueType> sumfVar) const;

  /// Emit ONE super-block's integer core (the plain 4-bit nibble unpack into the
  /// element-ordered aux8[256] scratch with NO bias; the STRUCTURED scalar 6-bit
  /// scale/min bit-dance via utmp/kmask; the nested sub-block loop applying the
  /// per-sub-block UINT6 scale in the i32 domain) at the current insertion point
  /// (INSIDE an already-open super-block loop body), and RETURN the
  /// per-super-block aux32[8] lvalue + the decoded scales/mins pointer. This is
  /// the genuinely-new q4_K super-block integer machinery, shared verbatim by
  /// K4a (which then vse8-stores the 16 scale/min bytes and vse32-stores the
  /// returned aux32) and K4b (which folds aux32 into the deferred fp32 sums and
  /// reads the mins for the min term). The emission mirrors _generic
  /// (quants.c:675-710) node-for-node: aux32 is RESET to zero each super-block,
  /// the unpack carries the entire q4 nibble->element layout (the dot then reads
  /// aux8 contiguously), and the 8-lane accumulation is integer (order-free).
  /// RETURNS the aux32 lvalue VARIABLE (not a load), so each caller can
  /// materialize the load at its own statement position (K4a's emitted C is
  /// byte-pinned by its ssh-rvv artifact).
  /// `scaleMinOutput`/`ib` are non-null ONLY for K4a (which exposes the 16
  /// decoded scale/min bytes through a uint8_t* output): when set, the helper
  /// vse8-stores the bytes IN-PLACE right after the bit-dance (before the aux32
  /// init), exactly where K4a has it today, so K4a's emitted node sequence stays
  /// byte-identical. K4b passes nullptr (no scale/min output) so the store is
  /// elided and the helper's order is (A unpack)(B bit-dance)(C sub-block dot).
  Q4_KCoreResult emitQ4_KSuperBlockAux32Core(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base,
      mlir::TypedValue<emitc::ArrayType> utmpArray, mlir::Value scaleMinOutput,
      mlir::Value ib) const;

  /// Emit the ggml ggml_vec_dot_q4_K_q8_K INTEGER CORE (the q4_K K4a increment)
  /// for one tcrv_rvv.q4_k_q8_k_aux_partial op as fully STRUCTURED emitc nodes
  /// (I5; no verbatim C-control-flow blob, no raw() -- including the 6-bit
  /// scale/min bit-dance, which is scalar emitc.bitwise_and/_or/_left_shift/
  /// _right_shift). It reproduces ggml _generic's per-super-block INTEGER state
  /// EXACTLY right before the fp32 d/dmin fold -- (1) aux32[8] (the per-sub-block
  /// uint6-scaled i32 accumulator) and (2) the 8 decoded 6-bit scales + 8 decoded
  /// 6-bit mins -- byte-exact, and writes both through the two output pointers
  /// (NO fp32 fold, NO min term -- that is K4b):
  ///   size_t nb = n / 256;  int8_t aux8[256];  uint32_t utmp[4];
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*144;  const uint8_t *yb = vy + ib*292;
  ///     // (A) unpack 256 weights into aux8 (element-ordered, plain 4-bit, NO
  ///     //     bias): FOUR 64-element chunks, each vand 0x0F (low nibble ->
  ///     //     a[0..31]) + vsrl 4 (high nibble -> a[32..63]).
  ///     // (B) the 6-bit scale/min bit-dance (utmp/kmask), scalar emitc bitwise:
  ///     //     read 12 packed bytes as 3 uint32, shuffle into utmp[4], store the
  ///     //     16 bytes [scales[0..7], mins[0..7]] through scalemin_out+ib*16.
  ///     // (C) per-sub-block uint6-scaled i32 dot into the 8-lane aux32 vector:
  ///     vint32m2_t aux32 = vmv_v_x_i32m2(0, 8);   // RESET per super-block
  ///     for (size_t js = 0; js < 8; js += 1) {
  ///       int32_t scale = scales[js];             // uint8 ZERO-EXTENDED scalar
  ///       // four quarters of 8: vwmul i8xi8->i16 then vwmacc.vx i32 += scale*i16
  ///     }
  ///     vse32_v_i32m2(aux32_out + ib*8, aux32, 8); // store aux32[8] state
  ///   }
  /// The unpack carries the entire q4 nibble->element layout (the dot then reads
  /// aux8 contiguously exactly as _generic's `a += 8`); the per-sub-block scale is
  /// UINT8 zero-extended (not q6_K's sign-extended int8) and applied in the i32
  /// domain (vwmacc.vx); the 8-lane aux32 accumulation is order-free (integer
  /// add). The block-format facts are the op's typed attrs (I4 mirror); the
  /// emission is the op's fixed structure. A SEPARATE core from q6_K's
  /// emitQ6_KSuperBlockAux32Core (different unpack, no qh, no bias, uint scale) so
  /// q6_K's emitted bytes stay byte-identical (additive).
  mlir::LogicalResult emitQ4_KQ8_KAux32Partial(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the Track B q4_K BRICK 1 op (tcrv_rvv.q4_k_nibble_unpack) as fully
  /// STRUCTURED emitc nodes: DECLARE the function-scoped int8_t aux8[256] scratch
  /// and fill it with ONE super-block's Region-A plain 4-bit nibble unpack (the
  /// shared emitQ4_KPlainNibbleUnpack helper, byte-identical to the monolithic
  /// q4_K integer core's Region A). The weight base operand IS the super-block
  /// pointer (single super-block; NO nb = n/256 loop, NO bit-dance, NO dot, NO
  /// fold -- those are deferred bricks). Binds the op's i32 m1 token to a zero
  /// literal (the unpack writes aux8 as a side effect; the token has no live use).
  mlir::LogicalResult emitQ4_KNibbleUnpack(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the Track B q4_K BRICK 2 op (tcrv_rvv.q4_k_scale_min_bit_dance) as
  /// fully STRUCTURED emitc nodes: DECLARE the function-scoped uint32_t utmp[4]
  /// scratch and fill it with ONE super-block's Region-B 6-bit scale/min
  /// bit-dance (the shared emitQ4_KScaleMinBitDanceCore helper, byte-identical to
  /// the monolithic q4_K integer core's Region B), then emit the store_scale_min
  /// observable (vse8 the 16 decoded [scales,mins] bytes through a local
  /// uint8_t scale_min[16] sink) so the lit has output to CHECK. The weight base
  /// operand IS the super-block pointer (single super-block; NO nb = n/256 loop,
  /// NO nibble unpack, NO dot, NO fold -- those are BRICK 1 / deferred bricks).
  /// Binds the op's i32 m1 token to a zero literal (the bit-dance writes utmp/
  /// scale_min as a side effect; the token has no live use).
  mlir::LogicalResult emitQ4_KScaleMinBitDance(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the Track B q4_K BRICK 3 op (tcrv_rvv.q4_k_scaled_dot) as fully
  /// STRUCTURED emitc nodes: derive the integer_core_lmul widening chain (the
  /// SAME detail::deriveWideningChain the monolithic q4_K core uses), build the
  /// integer-core context with the WIDE Region-C MAC types, then call the shared
  /// emitQ4_KScaledDotIntoAux32 helper (byte-identical to the monolithic q4_K
  /// integer core's Region C: the per-sub-block uint6-scaled i32 dot into the
  /// 8-lane aux32 + the C-tail integer fold-back) on the aux8 / scales / q8 ABI
  /// input pointer operands, and emit the store_aux32 observable (vse32 the
  /// canonical-8 aux32 through a local int32_t aux32_out[8] sink) so the lit has
  /// output to CHECK. At the default mf2 anchor NO fold-back appears (the running
  /// aux32 is already canonical-8); at the wide m1/m2 anchors the fold-back
  /// vslidedown/vadd/vget sequence appears -- the q4_K capability flip. The aux8 /
  /// scales / q8 operands are ABI INPUT pointers (single super-block; NO nb =
  /// n/256 loop, NO nibble unpack, NO bit-dance, NO MIN term, NO fp32 fold --
  /// those are BRICK 1 / BRICK 2 / deferred bricks). Binds the op's i32 m1 token
  /// to a zero literal (the dot writes aux32_out as a side effect; no live use).
  mlir::LogicalResult emitQ4_KScaledDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the Track B q4_K BRICK 4 op (tcrv_rvv.q4_k_min_term) as fully
  /// STRUCTURED emitc nodes: declare the op's OWN scalar `float sumf = 0.0f`
  /// accumulator + a local sink, load the fp32 activation scale dy once (the
  /// monolith's fold_activation_d, here the op's own setup), then call the SAME
  /// two shared MIN-term helpers the monolithic q4_K/q5_K block dot calls --
  /// emitQ4_KMinTermBsumsDot (the int16 bsums load + the scalar integer reduction
  /// sumi = sum(bsums * mins), reading the activation + decoded-scales pointers)
  /// and emitQ4_KMinTermSubtract (the fp16 dmin read + the `sumf -= dmin * sumi`
  /// emitc.expression, reading the weight pointer + dy) -- and store the resulting
  /// sumf through a local float sink as the observable so the lit has output to
  /// CHECK. The weight / scales / activation operands are ABI INPUT pointers
  /// (single super-block; NO nb = n/256 loop, NO nibble unpack, NO bit-dance, NO
  /// scaled dot, NO positive fold -- those are BRICK 1 / BRICK 2 / BRICK 3 /
  /// deferred bricks). Binds the op's i32 m1 token to a zero literal (the term
  /// writes sumf as a side effect; no live use).
  mlir::LogicalResult emitQ4_KMinTerm(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the Track B q4_K BRICK 6 op (tcrv_rvv.q4_k_sums_fold_scale_d) as fully
  /// STRUCTURED emitc nodes: declare the op's OWN 8-lane fp32 `vfloat32m2_t sums =
  /// vfmv_v_f_f32m2(0.0f, 8)` accumulator + a local float sums_out[8] sink,
  /// vle32-load the BRICK 3 canonical-8 aux32 from the ABI int32 pointer into a
  /// vint32m2_t lvalue, load the fp32 activation scale dy once (the monolith's
  /// fold_activation_d, here the op's own setup), then call the SAME shared
  /// positive-fold helper the monolithic q4_K/q5_K block dot calls --
  /// emitQ4_KSumsFoldScaleD (the fp16 d read + the vfcvt/vfmul/vfadd 8-lane
  /// sequence, reading the weight pointer + aux32 lvalue + dy) -- and store the
  /// resulting sums via vse32 through the local sink as the observable so the lit
  /// has output to CHECK. The weight / aux32 / activation operands are ABI INPUT
  /// pointers (single super-block; NO nb = n/256 loop, NO nibble unpack, NO
  /// bit-dance, NO scaled dot, NO MIN term, NO post-loop horizontal fold -- those
  /// are BRICK 1 / BRICK 2 / BRICK 3 / BRICK 4 / deferred bricks). Binds the op's
  /// i32 m1 token to a zero literal (the fold writes sums_out as a side effect; no
  /// live use).
  mlir::LogicalResult emitQ4_KSumsFoldScaleD(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the Track B q4_K BRICK 7 op (tcrv_rvv.q4_k_horizontal_fold) as fully
  /// STRUCTURED emitc nodes: declare the op's OWN 8-lane fp32 `vfloat32m2_t sums`
  /// accumulator (vle32-loaded from the ABI const float * source so the fold has
  /// observable input) + a `float sumf` accumulator (seeded 0.0f) + a `float
  /// sums8[8]` scratch + a local `float sumf_out[1]` sink, then call the SAME shared
  /// horizontal-fold helper the monolithic q4_K block dot calls --
  /// emitQ4_KHorizontalFold (the vse32 of sums into sums8 + the SEQUENTIAL 8-add
  /// `sumf += sums8[l]`, returning the final sumf) -- and store the resulting sumf
  /// into the local sink as the observable so the lit has output to CHECK. The sums
  /// operand is an ABI INPUT pointer (single super-block; NO nb = n/256 loop, NO
  /// nibble unpack, NO bit-dance, NO scaled dot, NO MIN term, NO positive fold --
  /// those are BRICK 1 / BRICK 2 / BRICK 3 / BRICK 4 / BRICK 6 / the deferred BRICK
  /// 5 loop). Binds the op's i32 m1 token to a zero literal (the fold writes
  /// sumf_out as a side effect; no live use).
  mlir::LogicalResult emitQ4_KHorizontalFold(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Shared context for the q2_K super-block SCALAR integer core + scalar fold
  /// (the byte-exact anchor extracted from the retired monolith
  /// emitQ2_KQ8_KBlockDot so the front-door-constructed typed super-block loop
  /// lowers byte-identically). Carries the EmitC types + the q2_K block-format
  /// facts the 2-bit unpack / plain-nibble scale-min / per-sub-block scalar dot /
  /// scalar fold need. i32Type == i32ImmType == the "int" opaque (kept as separate
  /// fields to mirror the monolith's local names). The fold facts (weight d @80 /
  /// dmin @82 fp16 offsets, activation d @0) are q2_K constants carried here so a
  /// caller with no separate fold brick (the loop form) reads them from the same
  /// place the monolith did.
  struct Q2_KIntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type i32Type;     // !emitc.opaque<"int">
    mlir::Type i32ImmType;  // !emitc.opaque<"int"> (immediate operands)
    mlir::Type floatType;   // !emitc.opaque<"float">
    mlir::Type u8m2Type;
    mlir::Type i8m2Type;
    mlir::Type i8m1Type;
    mlir::Type i16m2Type;
    mlir::Type i32m1Type;
    mlir::Type i8ElemType;
    mlir::Type i8PtrType;
    mlir::Type u8PtrType;
    mlir::Type constU8Type;
    mlir::Type constI16Type;
    mlir::Type constI16PtrType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    llvm::StringRef fp16ReadCallee;
    int64_t subBlock;             //  16
    int64_t numSubBlocks;         //  16
    int64_t qk;                   // 256
    int64_t scalesOffset;         //   0
    int64_t qsOffset;             //  16
    int64_t q8Offset;             //   4
    int64_t bsumsOffset;          // 260
    int64_t weightDOffset;        //  80 (fp16 x.d)
    int64_t weightDminOffset;     //  82 (fp16 x.dmin)
    int64_t activationDOffset;    //   0 (fp32 y.d)
  };

  /// Emit ONE q2_K super-block's INTEGER CORE (the 2-bit weight unpack into the
  /// element-ordered aux8[256] scratch, then the nested 16-sub-block loop applying
  /// the PLAIN uint4-nibble per-sub-block scale/min in the i32 domain) at the
  /// current insertion point (INSIDE an already-open super-block loop body), and
  /// RETURN the two per-super-block SCALAR integer-state lvalues (isum, summs). It
  /// mirrors _generic (quants.c:514-560) node-for-node: the 2-bit unpack carries
  /// the entire qs->element permutation (the dot then reads aux8 contiguously), the
  /// per-sub-block scale/min are the SIMPLE 4-bit nibbles of the direct scales[16]
  /// bytes (NO utmp/kmask bit-dance), and the per-super-block isum/summs are RESET
  /// to zero each super-block. This is the byte-exact anchor shared by the retired
  /// monolith and the typed super-block SCALAR-accumulator loop lowering. Returns
  /// the isum/summs lvalue VARIABLES (not loads) so the caller materializes the
  /// fold's loads at its own statement position.
  std::pair<mlir::TypedValue<emitc::LValueType>,
            mlir::TypedValue<emitc::LValueType>>
  emitQ2_KSuperBlockIntegerCore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q2_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base) const;

  /// Emit the q2_K per-super-block SCALAR fp32 fold `sumf += dall*isum -
  /// dmin*summs` (dy = *(const float *)(yb+0), dall = fp16(xb+80)*dy, dmin =
  /// fp16(xb+82)*dy) as ONE emitc.expression, byte-identical to the retired
  /// monolith. Reads the isum/summs integer states from their lvalues and updates
  /// the carried `sumf` scalar accumulator lvalue in place. Shared by the retired
  /// monolith and the typed SCALAR-accumulator loop lowering.
  void emitQ2_KScalarFold(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q2_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::LValueType> sumfVar,
      mlir::TypedValue<emitc::LValueType> isumVar,
      mlir::TypedValue<emitc::LValueType> summsVar) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_tq2_0_q8_K block kernel (the TERNARY
  /// {-1,0,+1} TriLM coverage rung) for one tcrv_rvv.tq2_0_q8_k_block_dot op as
  /// fully STRUCTURED emitc nodes (I5; no verbatim C-control-flow blob, no
  /// raw()). tq2_0 REUSES q2_K's 2-bit weight unpack VERBATIM but is genuinely
  /// SIMPLER -- NO scales, NO per-sub-block scale, NO min, NO dmin, NO bsums --
  /// mirroring _generic (quants.c:482-511) line-for-line so byte-exactness is
  /// by construction:
  ///   float sumf = 0.0f;                                        // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*66;  const uint8_t *yb = vy + ib*292;
  ///     // (A) the 2-bit ternary unpack: for each 32-byte qs chunk (k in 0..1)
  ///     //     and each shift in {0,2,4,6}, aux8[128*k + 32*(shift/2) + l] =
  ///     //     ((qs[k*32+l] >> shift) & 3) - 1 (the 32 lanes l) -- u8m2 load +
  ///     //     vsrl + vand + u8->i8 reinterpret + vadd.vx(-1) + vse8. The `-1`
  ///     //     ternary bias is folded PER ELEMENT into the unpack (mirrors
  ///     //     _generic's `(((qs>>shift)&3) - 1)`); the aux8 ordering pairs
  ///     //     contiguously with q8 (aux8[i] <-> q8[i]).
  ///     int sumi = 0;
  ///     for (size_t s = 0; s < 16; ++s)                         // 16x16 elems
  ///       sumi += vmv_x_s(vwredsum(vwmul(q8[16s..], aux8[16s..]), seed0));
  ///     // (B) the single-scale SCALAR fp32 fold, ONE C statement:
  ///     float d = *(const float *)(yb + 0) * (float)*(const _Float16 *)(xb+64);
  ///     sumf += (float)sumi * d;
  ///   }
  ///   *s = sumf;
  /// The integer side is order-free (associative int add) so the per-sub-block
  /// 16-lane reduce is summed into a SINGLE per-super-block scalar `sumi` (NO
  /// per-sub-block scale multiply -- tq2_0 has none); the ONLY pinned order is
  /// the SCALAR fp32 fold `sumf += (float)sumi * d` carried in super-block
  /// order, with `d = y.d * fp16(x.d)` as its OWN product so the association
  /// matches _generic (quants.c:506-508). The fold is ONE emitc.expression so
  /// it renders as ggml's single C statement and tracks the contraction. The
  /// block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitTQ2_0Q8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_tq1_0_q8_K block kernel (the BASE-3
  /// TriLM coverage rung -- the LAST of the 24 ggml dot kernels) for one
  /// tcrv_rvv.tq1_0_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-control-flow blob, no raw()). tq1_0 is the base-3-PACKED sibling
  /// of tq2_0: instead of a 2-bit field shift it recovers each trit by a
  /// power-of-three multiply + the uint8 wrap (5 trits per qs byte, 4 per qh
  /// byte). It reuses tq2_0's dot + fold + store VERBATIM (a single i32 `sumi`
  /// over the super-block, then `sumf += (float)sum * (fp16(x.d) * y.d)`); only
  /// the unpack differs. It mirrors _generic (quants.c:430-480) line-for-line so
  /// byte-exactness is by construction:
  ///   float sumf = 0.0f;                                        // ONCE
  ///   const uint8_t pow3[6] = {1,3,9,27,81,243};
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*54;  const uint8_t *yb = vy + ib*292;
  ///     // (A) the BASE-3 trit unpack into aux8[256] (in q8 index order):
  ///     //  - main (j=0): l 0..4 over qs[0..31] -> aux8[l*32 + m] (32 lanes)
  ///     //  - tail (j=32): l 0..4 over qs[32..47] -> aux8[160 + l*16 + m] (16)
  ///     //  - qh:          l 0..3 over qh[0..3]  -> aux8[240 + l*4 + j] (4)
  ///     // each trit: q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1.
  ///     // The 8-bit `vmul.vx` is the mandatory uint8 wrap; the widening `*3`
  ///     // + `>>8` reads the high base-3 digit; `vadd.vx -1` is the ternary
  ///     // bias. aux8[i] pairs contiguously with q8[i].
  ///     int sumi = 0;
  ///     for (size_t s = 0; s < 16; ++s)                         // 16x16 elems
  ///       sumi += vmv_x_s(vwredsum(vwmul(q8[16s..], aux8[16s..]), seed0));
  ///     // (B) the single-scale SCALAR fp32 fold, ONE C statement:
  ///     float d = (float)*(const _Float16 *)(xb+52) * *(const float *)(yb+0);
  ///     sumf += (float)sumi * d;
  ///   }
  ///   *s = sumf;
  /// The pow3 multiply must NOT be fused into the widening multiply (that skips
  /// the mod-256 wrap and is wrong for byte*pow3 >= 256). pow3[l] is emitted as
  /// a per-l literal. The integer side is order-free (associative int add); the
  /// ONLY pinned order is the SCALAR fp32 fold, with `d = fp16(x.d) * y.d` as
  /// its OWN product so the association matches _generic (quants.c:476). The
  /// fold is ONE emitc.expression so it renders as ggml's single C statement.
  /// The block-format facts (stride 54, qs @0, qh @48, d @52) are the op's typed
  /// attrs (I4 mirror).
  mlir::LogicalResult emitTQ1_0Q8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Value-level emission of the ggml offset-binary asymmetric i4xi8
  /// decode/product chain, factored so BOTH the standalone
  /// tcrv_rvv.packed_i4_offset_binary_x_i8_product op (INC-1) AND the block-dot
  /// inner loop (INC-2a) drive the SAME structured nodes. Returns the i16
  /// widening product value (low + high halves) for the given already-mapped
  /// i8mf4 weight/activation values, their EmitC types, and the active VL:
  ///   w_xor   = vxor_vx_i8mf4(weight, 0x88, vl);  // offset-binary->two's-compl
  ///   w_low   = vsll_vx_i8mf4(w_xor, 4, vl);
  ///   v0      = vsra_vx_i8mf4(w_low, 4, vl);       // low nibble i8
  ///   v1      = vsra_vx_i8mf4(w_xor, 4, vl);       // high nibble i8
  ///   product = vwmul_vv_i16mf2(v0, actLow, vl);   // low half  <-> q8[0..15]
  ///   product = vwmacc_vv_i16mf2(product, v1, actHigh, vl); // + high half
  /// Pure node construction; no string plan read.
  mlir::FailureOr<mlir::Value> emitOffsetBinaryDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weight, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value bodyVL, mlir::Type srcEmitC, mlir::Type resultEmitC,
      llvm::StringRef srcDtype, llvm::StringRef srcLmul, unsigned resSEW,
      llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
      llvm::StringRef role) const;

  /// The DECODE half of the offset-binary asymmetric i4xi8 chain, factored so
  /// the G1 GEMM tile (INC-14) can HOIST it above the M-column loop and reuse
  /// the decoded nibble lanes across all M activation columns (weight-decode
  /// reuse). Returns the (v0, v1) signed-i8 low/high nibble lanes:
  ///   w_xor = vxor_vx(weight, 0x88, vl);   // offset-binary -> two's-complement
  ///   w_low = vsll_vx(w_xor, 4, vl);
  ///   v0    = vsra_vx(w_low, 4, vl);        // low nibble i8 (sign-extended)
  ///   v1    = vsra_vx(w_xor, 4, vl);        // high nibble i8 (sign-extended)
  /// The four nodes are byte-identical to the leading four nodes of the combined
  /// emitter, so a back-to-back decode+product caller is unchanged.
  std::pair<mlir::Value, mlir::Value> emitOffsetBinaryDecodeValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weight, mlir::Value bodyVL, mlir::Type srcEmitC,
      llvm::StringRef srcDtype, llvm::StringRef srcLmul, llvm::StringRef opName,
      llvm::StringRef role) const;

  /// The PRODUCT half of the offset-binary asymmetric i4xi8 chain, factored from
  /// the decode so the G1 GEMM tile (INC-14) can replay it per activation column
  /// against the SAME hoisted decoded v0/v1 lanes. Returns the i16 widening
  /// product (low + high halves):
  ///   product = vwmul_vv(v0, actLow, vl);                // low <-> q8[0..15]
  ///   product = vwmacc_vv(product, v1, actHigh, vl);     // + high half
  /// The two nodes are byte-identical to the trailing two nodes of the combined
  /// emitter, so a back-to-back decode+product caller is unchanged.
  mlir::FailureOr<mlir::Value> emitOffsetBinaryProductFromDecodedValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value v0, mlir::Value v1, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value bodyVL, mlir::Type resultEmitC, unsigned resSEW,
      llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
      llvm::StringRef role) const;

  /// The UNSIGNED-nibble asymmetric i4xi8 decode/product for ONE strip -- the
  /// Family-B (q4_1) variant of emitOffsetBinaryDecodeProductValue, added cleanly
  /// alongside it (the offset-binary one is byte-untouched). ggml q4_1 decodes
  /// the nibbles as UNSIGNED [0,15] (NO offset-binary `-8` bias, NO XOR-0x88):
  ///   x_a = vand_vx_u8<L>(weightU8, 0x0F, vl);   // low nibble  [0,15]
  ///   x_l = vsrl_vx_u8<L>(weightU8, 0x04, vl);   // high nibble [0,15] (LOGICAL)
  ///   v0  = vreinterpret_v_u8<L>_i8<L>(x_a);     // value-identity for 0..15
  ///   v1  = vreinterpret_v_u8<L>_i8<L>(x_l);
  ///   product = vwmul_vv_i16<W>(v0, actLow, vl);            // low <-> q8[0..15]
  ///   product = vwmacc_vv_i16<W>(product, v1, actHigh, vl); // + high half
  /// The weight lane MUST be u8 (vsrl is a LOGICAL shift; there is no
  /// vsrl_vx_i8); the reinterprets are free (value-identity for 0..15). Pure node
  /// construction; no string plan read.
  mlir::FailureOr<mlir::Value> emitUnsignedNibbleDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weightU8, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value bodyVL, mlir::Type srcI8EmitC, mlir::Type srcU8EmitC,
      mlir::Type resultEmitC, llvm::StringRef srcLmul, unsigned resSEW,
      llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
      llvm::StringRef role) const;

  /// The 5-BIT offset-binary asymmetric i5xi8 decode/product for ONE strip -- the
  /// Family-A (q5_0) variant of the nibble decoders, added cleanly alongside the
  /// offset-binary (q4_0) and unsigned (q4_1) ones (both byte-untouched). ggml
  /// q5_0 reconstructs each weight from a 4-bit nibble PLUS a per-element 5th high
  /// bit packed in a separate 32-bit qh field, then applies the offset-binary
  /// `-16` bias: for element i (low half) value = `(int8_t)(((qs[i]&0x0F) |
  /// (((qh>>i)&1)<<4)) - 16)`, for element i (high half) value =
  /// `(int8_t)(((qs[i]>>4) | (((qh>>(i+16))&1)<<4)) - 16)`. This emitter realizes
  /// that, for a strip covering element indices [c, c+vl) within the half-block:
  ///   xLow  = vand_vx_u8(weightU8, 0x0F, vl);  // low nibble  [0,15]
  ///   xHigh = vsrl_vx_u8(weightU8, 0x04, vl);  // high nibble [0,15] (LOGICAL)
  ///   // per-lane 5th bit: a vid+c shift vector selects each lane's qh bit.
  ///   idx   = vid_v_u16<W>(vl);                 // 0..vl-1
  ///   sh    = vadd_vx_u16<W>(idx, c, vl);       // absolute element index
  ///   lowB  = vmv_v_x_u16<W>(qhLow16, vl);      // qh & 0xFFFF broadcast
  ///   lowB  = vsrl_vv_u16<W>(lowB, sh, vl);     // shift each lane's bit to b0
  ///   lowB  = vand_vx_u16<W>(lowB, 1, vl);      // isolate {0,1}
  ///   lowB  = vsll_vx_u16<W>(lowB, 4, vl);      // place at bit 4 -> {0,16}
  ///   lowHB = vncvt_x_x_w_u8<core>(lowB, vl);   // narrow u16->u8
  ///   (same for the high half off qhHigh16 = qh>>16)
  ///   fiveLow  = vor_vv_u8<core>(xLow,  lowHB, vl);  // [0,31]
  ///   fiveHigh = vor_vv_u8<core>(xHigh, hiHB, vl);
  ///   v0 = vsub_vx_i8<core>(vreinterpret(fiveLow),  16, vl);  // [-16,15]
  ///   v1 = vsub_vx_i8<core>(vreinterpret(fiveHigh), 16, vl);
  ///   product = vwmul_vv_i16<W>(v0, actLow, vl);            // low <-> q8[0..15]
  ///   product = vwmacc_vv_i16<W>(product, v1, actHigh, vl); // + high half
  /// The shift uses `vid + c` (the absolute element index), so the per-strip 5th
  /// bit aligns for ANY chunk offset c (the robust mf4 multi-strip path) and ANY
  /// VLEN. The high half reads `qh >> 16` (bits 16..31), the low half `qh & 0xFFFF`
  /// (bits 0..15). The bit extraction runs in u16 at the wide LMUL (the same one
  /// the widening product uses); the nibble/OR/bias chain runs in u8/i8 at the
  /// core LMUL. Pure node construction; no string plan read.
  ///
  /// `applyOffsetBias` selects the bias model so the q5_1 (Family-B, scale+MIN,
  /// UNSIGNED 5-bit weight) kernel SHARES this exact unpack + 5th-bit injection:
  /// q5_1's ggml reconstruction is `(qs & 0xF) | ((qh>>j)<<4)&0x10` for the low
  /// half and `(qs>>4) | (qh>>(j+12))&0x10` for the high half -- the SAME low
  /// bit j / high bit (j+16) selection q5_0 uses, just with NO `-16` (the bias
  /// lives in q5_1's separate per-block MIN scale, exactly like q4_1). With
  /// applyOffsetBias=false the final `vsub 16` is skipped (the unsigned q5 in
  /// [0,31] feeds the signed widening product directly, value-identity), so the
  /// q5_0 emission (applyOffsetBias=true) stays byte-identical and the q5_1
  /// emission inherits the whole 5-bit reconstruction without duplicating it.
  mlir::FailureOr<mlir::Value> emitFiveBitOffsetBinaryDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weightU8, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value qhLow16, mlir::Value qhHigh16, mlir::Value chunkOffset,
      mlir::Value bodyVL, mlir::Type srcI8EmitC, mlir::Type srcU8EmitC,
      mlir::Type wideU16EmitC, mlir::Type resultEmitC, llvm::StringRef srcLmul,
      llvm::StringRef wideLmul, unsigned resSEW, llvm::StringRef resLmul,
      llvm::StringRef resDtype, llvm::StringRef opName, llvm::StringRef role,
      bool applyOffsetBias = true) const;

  /// Emit the COMPLETE ggml ggml_vec_scale_f32 forward-pass op (the FIRST
  /// non-dot f32 elementwise family member) for one tcrv_rvv.ggml_vec_scale_f32
  /// op as fully STRUCTURED emitc nodes (I5; no verbatim C-string blob -- every
  /// value is a node in the IR graph):
  ///   for (size_t i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e32m<L>(n - i);    // emitc.sub + call_opaque
  ///     vfloat32m<L>_t ay = __riscv_vle32_v_f32m<L>(y + i, vl);
  ///     vfloat32m<L>_t ny = __riscv_vfmul_vf_f32m<L>(ay, v, vl);  // scalar bcast
  ///     __riscv_vse32_v_f32m<L>(y + i, ny, vl);       // in-place store back
  ///   }
  /// `y` is read AND written in place (the first forward-pass op whose single
  /// buffer is both input and output); `v` is the runtime f32 scalar broadcast
  /// into every lane. Byte-exactness to ggml's real op (vec.h:733-739) is
  /// UNCONDITIONAL: a bare per-lane fp32 multiply (no FMA -> -ffp-contract
  /// cannot bite; no cross-lane reduction -> LMUL/tail/strip-count are
  /// correctness-free). The LMUL is the bounded resource/scheduling knob
  /// (default m8, matching ggml). The intrinsics are emitc.call_opaque nodes
  /// (the one sanctioned opaque piece, exactly how the dot kernels emit theirs).
  mlir::LogicalResult emitGgmlVecScaleF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_compute_forward_rms_norm_f32 forward-pass op
  /// (non-fused, no weight; ops.cpp:3758-3817) for ONE row as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-string blob -- every value is a node):
  ///   double sum = 0.0;                          // ggml_float accumulator
  ///   for (size_t i = 0; i < ne00; ++i) {        // SCALAR ascending fold
  ///     float p = x[i] * x[i];                    // f32 product (one f32 round)
  ///     sum = sum + (double)p;                    // widen-to-double then add
  ///   }
  ///   float mean  = (float)(sum / (double)ne00);  // divide in double, cast after
  ///   float scale = 1.0f / sqrtf(mean + eps);     // f32 add/sqrtf/reciprocal
  ///   for (size_t i = 0; i < ne00; i += vlmax) {  // VECTORIZED normalize strip
  ///     size_t vl = __riscv_vsetvl_e32m<L>(ne00 - i);
  ///     vfloat32m<L>_t vx = __riscv_vle32_v_f32m<L>(x + i, vl);
  ///     vfloat32m<L>_t vy = __riscv_vfmul_vf_f32m<L>(vx, scale, vl);
  ///     __riscv_vse32_v_f32m<L>(y + i, vy, vl);
  ///   }
  /// BYTE-EXACTNESS to ggml depends on the reduction METHOD, not a tolerance:
  /// ggml folds Sx^2 in `ggml_float` (= double), SCALAR, in strict ASCENDING
  /// index order (ops.cpp:3791-3795 -- explicitly NOT vectorized). The cast chain
  /// is load-bearing: each product rounds in f32 first, is WIDENED to double,
  /// then accumulated in double (NOT (double)x*(double)x). The f32->double widen
  /// between the f32 product and the double add is also an FMA barrier (different
  /// types -> -ffp-contract cannot fuse). The mean divides in double and casts to
  /// f32 AFTER; the scale is sqrtf-then-reciprocal in f32 (NOT a fast-rsqrt).
  /// Only the final normalize is vectorized -- a bare per-lane vfmul_vf (no FMA,
  /// no reduction), byte-exact at any LMUL because every lane is multiplied by
  /// the same scalar `scale`. The reduction is emitted as STRUCTURED scalar emitc
  /// nodes (variable/load/mul/cast/add/assign), NOT a raw string.
  mlir::LogicalResult emitGgmlRmsNormF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit ggml's EXACT vectorized minimax exp polynomial `ggml_v_expf_m2`
  /// (vec.h:1324-1360) as a node-for-node STRUCTURED emitc chain, returning the
  /// `e^X` result as a `vfloat32m2_t` value. `X` is the already-computed m2 f32
  /// argument vector (silu feeds `vfneg(x)`; soft_max feeds `vfsub_vf(x, max)`),
  /// `bodyVL` the active strip VL.
  ///
  /// This is the ONE shared replication of ggml's exp polynomial, factored out
  /// of the F5 silu emitter so the F5b soft_max emitter consumes the IDENTICAL
  /// chain (the byte-exactness invariant: both emit the SAME instructions ggml's
  /// source compiles to). The slow-path overflow/underflow vmerge value graph is
  /// emitted UNCONDITIONALLY -- ggml's `if (!vcpop_m(c))` is a pure performance
  /// short-circuit whose fast path `k + j*k` equals the slow path's c-false lane
  /// `k + k*j` bit-for-bit (fp multiply commutes; RISC-V yields the canonical NaN
  /// regardless of operand order), so there is no data-dependent branch. Pinned
  /// at m2 (the m2-tied vbool16_t mask / vuint32m2_t reinterpret types).
  mlir::Value emitGgmlVExpfM2(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value X,
                              mlir::Value bodyVL, mlir::Type sizeType,
                              llvm::StringRef opName, llvm::StringRef role) const;

  /// Emit the COMPLETE ggml ggml_vec_silu_f32 forward-pass op (the FFN
  /// activation y[i] = x[i]*sigmoid(x[i]), sigmoid(x) = 1/(1+e^{-x});
  /// vec.cpp:380 + ggml_v_silu_m2 vec.h:1363) as fully STRUCTURED emitc nodes
  /// (I5; no verbatim C-string blob -- every value is a node):
  ///   for (size_t i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e32m2(n - i);
  ///     vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + i, vl);
  ///     vfloat32m2_t vy = ggml_v_silu_m2(vx, vl);     // expanded node-for-node
  ///     __riscv_vse32_v_f32m2(y + i, vy, vl);
  ///   }
  /// silu = vfneg(x) -> ggml_v_expf_m2 -> vfadd 1.0f -> vfdiv(x, 1+exp).
  ///
  /// BYTE-EXACTNESS to ggml's REAL vectorized silu hinges on replicating
  /// ggml_v_expf_m2 (vec.h:1324-1360) node-for-node -- a fully vectorized minimax
  /// exp polynomial built ENTIRELY from __riscv_v intrinsics (NO libm expf). Each
  /// intrinsic is one emitc.call_opaque node (the one sanctioned opaque seam, as
  /// the dot kernels emit theirs) with the IDENTICAL magic-constant bit patterns
  /// (0x1.8p23f, 0x1.715476p+0f, 0x1.62e4p-1f, 0x1.7f7d1cp-20f, the degree-5
  /// polynomial coefficients, 0x3f800000, 0x82000000/0x7f000000/126.0f/192.0f).
  /// ggml's `if (!vcpop_m(c))` is a pure performance short-circuit -- the fast
  /// path k + j*k equals the slow path's c-false lane k + k*j bit-for-bit (fp
  /// multiply commutes; RISC-V yields the canonical NaN regardless of operand
  /// order) -- so the slow-path vmerge value graph is emitted UNCONDITIONALLY as
  /// a straight-line chain: same output bits for every input (normal, saturating
  /// tails, NaN/inf/denormal), no data-dependent branch. Pinned at m2 (matching
  /// ggml's vsetvl_e32m2 path and the m2-tied vbool16_t/vuint32m2_t types).
  mlir::LogicalResult emitGgmlVecSiluF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_soft_max_f32 forward-pass op (F5b: the
  /// attention softmax core y[i] = e^{x[i]-max}, RETURNING the f64 sum
  /// Sum_i e^{x[i]-max}; vec.cpp:531 + the __riscv_v path vec.cpp:584-592) as
  /// fully STRUCTURED emitc nodes (I5; no verbatim C blob -- every value is a
  /// node). Faithful to ggml's BARE function: `max` is an INPUT, no normalize.
  ///   vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
  ///   for (size_t i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e32m2(n - i);
  ///     vfloat32m2_t val = ggml_v_expf_m2(
  ///         __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(x + i, vl), max, vl),
  ///         vl);                                   // shared exp polynomial
  ///     __riscv_vse32_v_f32m2(y + i, val, vl);
  ///     vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);  // f32->f64
  ///   }
  ///   return (double)__riscv_vfmv_f_s_f64m1_f64(vsum);
  ///
  /// The sum is the byte-exactness crux: ggml accumulates in `ggml_float` =
  /// DOUBLE via the WIDENING reduce vfwredusum_vs_f32m2_f64m1 into a SINGLE f64m1
  /// accumulator carried across strips (NOT F3's scalar-ascending fold). The
  /// f64m1 accumulator is a loop-carried opaque-vector lvalue (emitc.for has no
  /// iter_args), exactly as F3 carries its scalar double `sum`, but the value
  /// type is the opaque vector vfloat64m1_t. exp(x-max) reuses the SHARED
  /// node-for-node ggml_v_expf_m2 chain (emitGgmlVExpfM2), so y[] and each val
  /// are bit-identical to ggml's silu/soft_max. Returns the f64 sum value (the
  /// dispatch wraps it in the function's `return`).
  mlir::FailureOr<mlir::Value> emitGgmlVecSoftMaxF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml `quantize_row_q8_0` RVV-path forward-pass op (the F4
  /// f32 -> block_q8_0 activation quantizer; riscv/quants.c:32-71) for one
  /// tcrv_rvv.quantize_row_q8_0 op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph):
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     size_t vl = 32;                                    // = QK8_0 (one strip)
  ///     const float *xb = x + ib*32;
  ///     uint8_t *yb = vy + ib*34;                          // AoS block cursor
  ///     vfloat32m8_t v_x  = __riscv_vle32_v_f32m8(xb, vl);
  ///     vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(v_x, vl);
  ///     vfloat32m1_t tmp  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  ///     vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, tmp, vl);
  ///     float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
  ///     float d  = amax / 127.0f;                          // scalar f32
  ///     float id = 0.0f; if (d != 0.0f) { id = 1.0f / d; } // d ? 1/d : 0
  ///     *(_Float16 *)(yb + 0) = (_Float16)d;               // fcvt.h.s (rne)
  ///     vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);
  ///     vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);  // f32->i16 (rne)
  ///     vint8m2_t  vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);    // i16->i8 truncate
  ///     __riscv_vse8_v_i8m2(yb + 2, vs, vl);                // the 32 int8 qs
  ///   }
  /// BYTE-EXACTNESS to the DEPLOYED kernel matches ggml's EXACT RVV method, NOT
  /// the scalar `_ref` (which rounds with `roundf`, round-half-AWAY). The two
  /// cruxes: (1) vfncvt_x_f_w_i16m4 rounds with the dynamic frm = round-to-
  /// nearest-EVEN -- replicating that exact intrinsic inherits rne + the i8
  /// saturating clamp + every edge case (do NOT reason about rounding); (2) the
  /// board is __riscv_zfhmin, so GGML_CPU_FP32_TO_FP16(d) is the native
  /// (_Float16)d cast (fcvt.h.s, rne) -- a STRUCTURAL _Float16 store, not a
  /// software fp16 pack. The `id = d ? 1/d : 0` conditional is load-bearing (the
  /// all-zero block: amax=0 => d=0 must give id=0 so every q=0; a bare 1/d gives
  /// inf/NaN) -- emitted as a STRUCTURED emitc.cmp + emitc.if, NOT a raw string.
  /// The block-format facts (qk/stride/offsets) are the op's typed attrs (I4);
  /// the emission is the op's fixed structure. vl is hard-pinned to QK8_0=32 (one
  /// e32m8 strip per block, relying on Zvl128b => VLEN>=128, the same capability
  /// the q4_0 mb4-elided shape uses) -- replicating ggml's `size_t vl = QK8_0`.
  /// The intrinsic spellings are HARD-CODED callees (vfabs/vfredmax/vfncvt/vncvt),
  /// matching ggml's exact path -- never synthesized (a _rm/_tu suffix would
  /// change the rounding mode or fail to compile).
  mlir::LogicalResult emitGgmlQuantizeRowQ80(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_compute_forward_rope_f32 op for ONE head row,
  /// the GGML_ROPE_TYPE_NORMAL variant (the rope llama-2 / LLM_ARCH_LLAMA uses),
  /// as fully STRUCTURED emitc nodes (I5; no verbatim C-string blob):
  ///   float theta = theta_base;                        // the recurrence seed
  ///   size_t n_pairs = n_dims / 2;
  ///   for (size_t p = 0; p < n_pairs; ++p) {           // SCALAR per-pair loop
  ///     float cos_t = cosf(theta);  float sin_t = sinf(theta);  // scalar libm
  ///     const float *xp = (const float *)(x + 2*p);
  ///     float *yp = (float *)(y + 2*p);
  ///     float x0 = xp[0];  float x1 = xp[1];           // CONSECUTIVE pair
  ///     yp[0] = x0*cos_t - x1*sin_t;                   // two SEPARATE muls then
  ///     yp[1] = x0*sin_t + x1*cos_t;                   //   a sub/add (no FMA)
  ///     theta = theta * theta_scale;                   // iterative f32 recurrence
  ///   }
  /// BYTE-EXACTNESS has TWO axes (both stated honestly in the result):
  /// (1) cosf/sinf are SCALAR libm (one emitc.call_opaque each -- the sanctioned
  ///     opaque seam, NOT a raw string), so the angles are bit-exact vs ggml's
  ///     cache[] only when the SAME libm is linked (a libm-tolerance otherwise).
  ///     This differs from F5's exp (a vectorized polynomial we replicate); rope
  ///     is NOT a vectorized-transcendental problem.
  /// (2) The rotation x0*cos - x1*sin is a*b - c*d, an FP-contraction hazard.
  ///     ggml's rotation is a SINGLE C expression (ops.cpp:5808-5809), so each
  ///     output's a*b - c*d is GROUPED into ONE emitc.expression here (the F3
  ///     rms_norm emitc.expression FMA-fix discipline) -> ONE C statement
  ///     token-identical to ggml's -> clang makes the IDENTICAL contraction
  ///     decision under EVERY -ffp-contract mode, so the kernel is byte-exact vs
  ///     ggml under default/on/off/fast (NOT just off), independent of the build
  ///     flag. Separate-statement products would block intra-statement fusion and
  ///     diverge from ggml under the default `on`.
  /// theta_base (= pos as f32) and theta_scale (= powf(freq_base, -2/n_dims)) are
  /// PRECOMPUTED runtime f32 inputs, so the kernel makes no powf call -- the only
  /// libm calls are the per-pair cosf/sinf. The loop is SCALAR (not vectorized):
  /// cos/sin are scalar libm so the faithful structure IS ggml's scalar per-pair
  /// rotation (vectorizing would gather scalars into vectors with no exactness
  /// gain). The recurrence theta is a loop-carried emitc.variable lvalue +
  /// emitc.assign (emitc.for has no iter_args), exactly as F3's scalar-double sum.
  mlir::LogicalResult emitGgmlRopeNormF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  mlir::LogicalResult emitPackedI4OffsetBinaryXI8Product(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::PackedI4OffsetBinaryXI8ProductOp packed,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// The CODEBOOK-gather decode/product for ONE strip -- the FP4 / iq4_nl
  /// variant of emitOffsetBinaryDecodeProductValue (the offset-binary one is
  /// byte-untouched). The 4-bit nibble does NOT decode linearly; each nibble is
  /// an INDEX into the non-linear 16-entry int8 `table` (the broadcast values):
  ///   idx_low  = vand_vx_u8<L>(weightU8, 0x0F, vl);   // low index  [0,15]
  ///   idx_high = vsrl_vx_u8<L>(weightU8, 0x04, vl);   // high index [0,15]
  ///   v0 = vrgather_vv_i8<L>(table, idx_low,  vl);    // gather -> signed-i8
  ///   v1 = vrgather_vv_i8<L>(table, idx_high, vl);
  ///   product = vwmul_vv_i16<W>(v0, actLow, vl);            // low <-> q8[0..15]
  ///   product = vwmacc_vv_i16<W>(product, v1, actHigh, vl); // + high half
  /// The product tail REUSES emitOffsetBinaryProductFromDecodedValue (byte
  /// identical to the offset-binary sibling's two product nodes); only the
  /// codebook split + gather REPLACES the xor/sll/sra arithmetic decode. The
  /// node order is byte-identical to the monolithic codebook block-dot's
  /// per-strip core (RVVToEmitCCodebookFp4.cpp). Pure node construction; no
  /// string plan read.
  mlir::FailureOr<mlir::Value> emitCodebookGatherDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weightU8, mlir::Value table, mlir::Value actLow,
      mlir::Value actHigh, mlir::Value bodyVL, mlir::Type srcI8EmitC,
      mlir::Type srcU8EmitC, mlir::Type resultEmitC, llvm::StringRef srcLmul,
      unsigned resSEW, llvm::StringRef resLmul, llvm::StringRef resDtype,
      llvm::StringRef opName, llvm::StringRef role) const;

  /// codebook_table_broadcast{codebook,table_symbol} -> the structured const
  /// `static const int8_t <table_symbol>[16] = {...};` decl + the
  /// __riscv_vle8_v_i8<L> broadcast load of <table_symbol> at the 16-entry
  /// count (comment "codebook_table_load"). Registers the broadcast values vreg
  /// in valueMap for the gather to consume. Byte-identical to the monolithic
  /// codebook emitter's once-above-loop table broadcast.
  mlir::LogicalResult emitCodebookTableBroadcast(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::CodebookTableBroadcastOp table,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  mlir::LogicalResult emitCodebookGatherXI8Product(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::CodebookGatherXI8ProductOp gather,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// The UNSIGNED-nibble decode/product for ONE strip -- the q4_1 variant. The
  /// weight is an UNSIGNED u8/m1 vector; the two nibble lanes are split
  /// (vand_vx_u8<L> 0x0F low, vsrl_vx_u8<L> 0x04 high), reinterpreted u8->i8
  /// (value-identity for [0,15]), then fed the SAME asymmetric widening product
  /// the offset-binary sibling uses (vwmul <-> plain low activation, vwmacc <->
  /// plain high activation). NO xor-0x88 bias, NO codebook gather table. Routes
  /// to emitUnsignedNibbleDecodeProductValue; the signed-i8 EmitC type is built
  /// off the weight LMUL. Pure node construction; no string plan read.
  mlir::LogicalResult emitUnsignedNibbleXI8Product(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::UnsignedNibbleXI8ProductOp product,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// The FIVE-bit (nibble+qh) offset-binary decode/product for ONE strip -- the
  /// q5_0 variant. The weight is an UNSIGNED u8/m1 vector; the two nibble lanes are
  /// split (vand 0x0F / vsrl 0x04), each MERGED with the per-lane qh 5th bit
  /// (re-read from the block_five_bit_qh_source brick that defines the qh_source
  /// operand -- two aligned 16-bit halves off the brick's own qh_base +
  /// qh_byte_offset, the operand/source-driven anti-bypass), offset-binary biased
  /// `-16` into signed i8, then fed the SAME asymmetric widening product the
  /// siblings use. Routes to emitFiveBitOffsetBinaryDecodeProductValue with
  /// applyOffsetBias=true and a literal-0 chunkOffset (elided single-strip typed
  /// body). Pure node construction; no string plan read.
  mlir::LogicalResult emitFiveBitOffsetBinaryXI8Product(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::FiveBitOffsetBinaryXI8ProductOp product,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// The per-block qh 32-bit field SOURCE brick -- a GATE-ONLY edge (mirrors
  /// emitBlockFp16MinProduct): it materializes NOTHING into the valueMap; the
  /// naming five-bit product op re-reads the two qh halves from THIS brick's
  /// qh_base + qh_byte_offset. The handler only validates the operand mapping so
  /// the op-by-op walk does not fail on the op (the whole variant body is erased
  /// wholesale after the walk).
  mlir::LogicalResult emitBlockFiveBitQhSource(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::BlockFiveBitQhSourceOp qhSource,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// widening_macc(%lhs,%rhs,%acc,%vl){kind=signed_widening_macc_add} ->
  ///   v<rd><rl> r = __riscv_vwmacc_vv_<rd><rl>(acc, lhs, rhs, vl);
  /// The fused widening multiply-accumulate widens the narrower i16 source
  /// multiplicands and accumulates into the i32 accumulator vector. The C call
  /// order is (accumulator, lhs, rhs, vl); the intrinsic dtype/lmul derive from
  /// the RESULT (i32/m1) vector. The accumulator is an explicit loaded i32
  /// vector chunk (the per-chunk macc seed), not a scalar-carry cell.
  mlir::LogicalResult
  emitWideningMAcc(mlir::ConversionPatternRewriter &rewriter,
                   mlir::Location loc, tcrvrvv::WideningMAccOp macc,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const;

  /// widening_dot_reduce(%lhs,%rhs,%acc,%vl){kind=signed_widening_dot_reduce_add}
  /// is the scalar-carry-through-output dot product. The pre-loop seed
  /// (out[0]=acc[0]) and the lane-0 result store (out base, VL=1) are handled by
  /// the shared standalone-reduction machinery; the in-loop dataflow is:
  ///   v<rd>m1 p = __riscv_vwmul_vv_<rd>m1(lhs, rhs, vl);      // widened product
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1); // running seed
  ///   v<rd>m1 red = __riscv_vredsum_vs_<rd>m1_<rd>m1(p, seed, vl);
  /// The product widens i16/mf2 multiplicands into the i32/m1 result, so the
  /// vwmul/vredsum dtype/lmul derive from the RESULT (i32/m1) vector.
  mlir::LogicalResult
  emitWideningDotReduce(mlir::ConversionPatternRewriter &rewriter,
                        mlir::Location loc, tcrvrvv::WideningDotReduceOp dot,
                        llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                        mlir::Value outBuffer, mlir::Value bodyVL) const;

  /// masked_widening_dot_reduce(%mask,%lhs,%rhs,%acc,%vl) is the predicated
  /// scalar-carry dot product. The pre-loop seed / lane-0 store are shared. The
  /// in-loop dataflow zeroes inactive product lanes before reducing:
  ///   v<rd>m1 zero = vmv_v_x_<rd>m1(0, vl);                       // running vl
  ///   v<rd>m1 mp   = __riscv_vwmul_vv_<rd>m1_m(mask, lhs, rhs, vl); // masked
  ///   v<rd>m1 mrg  = __riscv_vmerge_vvm_<rd>m1(zero, mp, mask, vl); // 0 inactive
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1);
  ///   v<rd>m1 red = __riscv_vredsum_vs_<rd>m1_<rd>m1(mrg, seed, vl);
  mlir::LogicalResult emitMaskedWideningDotReduce(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedWideningDotReduceOp dot,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, mlir::Value outBuffer,
      mlir::Value bodyVL) const;

  /// The C scalar literal type the neutral splat constant prints as. The neutral
  /// is an integer literal (0 / INT_MAX / INT_MIN-shaped); emitc renders the
  /// literal verbatim, so an opaque int placeholder suffices for the SSA type.
  static mlir::Type resultIntScalarType(mlir::ConversionPatternRewriter &r);

  /// store(%abi,%val,%vl) -> ptr = base + i; __riscv_vse<sew>_v_<dtype><lmul>(...)
  /// When `extraOffset` is set, a SECOND pointer add is emitted after the
  /// `base + i` add: ptr2 = ptr + extraOffset (mirroring the legacy unrolled
  /// `v23 = base + i; v24 = v23 + priorVL` two-add form for the second u2 slice).
  /// Existing single-slice callers pass the default null and emit one add.
  mlir::LogicalResult
  emitStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::StoreOp store,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value inductionVar, mlir::Value storeVL,
            mlir::Value extraOffset = {}) const;

  /// broadcast_load(%abi,%vl) -> scalar = base[0]; __riscv_vmv_v_x_<dtype><lmul>
  /// The legacy materializer renders the RHS broadcast operand `rhs[0]` via an
  /// emitc.subscript + emitc.load reading the first element, then splats that
  /// scalar with vmv_v_x. Reproduced exactly so the rendered C carries the
  /// `const int32_t vN = base[0];` temp + the vmv_v_x splat.
  mlir::LogicalResult
  emitBroadcastLoad(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, tcrvrvv::BroadcastLoadOp broadcast,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value bodyVL) const;

  /// splat(%scalar,%vl) -> __riscv_vmv_v_x_<dtype><lmul>(scalar, vl). The scalar
  /// is a runtime ABI value mapped to a function parameter directly.
  mlir::LogicalResult
  emitSplat(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::SplatOp splat,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value bodyVL) const;

  /// compare(%lhs,%rhs,%vl){kind} ->
  ///   __riscv_v<cmp>_vv_<dtype><lmul>_b<maskbits>(lhs, rhs, vl) -> vbool<n>_t
  mlir::LogicalResult
  emitCompare(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              tcrvrvv::CompareOp compare,
              llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
              mlir::Value bodyVL) const;

  /// select(%mask,%true,%false,%vl) ->
  ///   __riscv_vmerge_vvm_<dtype><lmul>(false, true, mask, vl)
  /// vmerge keeps the FALSE vector on inactive lanes and the TRUE vector on
  /// active lanes, so the operand order is (false_vec, true_vec, mask, vl) --
  /// byte-identical to the legacy compare-select select step.
  mlir::LogicalResult
  emitSelect(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::SelectOp select,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const;

  /// mask_and(%a,%b,%vl){kind} ->
  ///   __riscv_vmand_mm_b<maskbits>(a, b, vl)
  /// composes two predicate masks of the same (sew, lmul) into one mask.
  mlir::LogicalResult
  emitMaskAnd(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              tcrvrvv::MaskAndOp maskAnd,
              llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
              mlir::Value bodyVL) const;

  /// dequantize(%source_i32,%scale,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy i32->f32 runtime-scale dequant sequence:
  ///   converted = __riscv_vfcvt_f_x_v_<dtype><lmul>(source, vl);
  ///   result    = __riscv_vfmul_vf_<dtype><lmul>(converted, scale, vl);
  /// The result vector type (f32) drives the intrinsic suffix; the scale is a
  /// runtime ABI float value mapped to a function parameter directly.
  mlir::LogicalResult
  emitDequantize(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 tcrvrvv::DequantizeOp dequantize,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const;

  /// The bare vfcvt+vfmul dequant chain shared by the clamp epilogue
  /// (`emitDequantize`) and the standalone load->dequantize->store body
  /// (`emitStandaloneDequantBody`):
  ///   converted = __riscv_vfcvt_f_x_v_<dtype><lmul>(source, vl);
  ///   result    = __riscv_vfmul_vf_<dtype><lmul>(converted, scale, vl);
  /// `source`/`scale` are the already-mapped EmitC values; the result vector
  /// type (f32) drives the intrinsic suffix. Sets valueMap[result] so the
  /// downstream store lookup resolves. Fails closed (notifyMatchFailure) on a
  /// non-f32 result or an unconvertible vector type so a malformed body rolls
  /// back to the legacy materializer unchanged (no mislower).
  mlir::LogicalResult
  emitDequantizeChain(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, tcrvrvv::DequantizeOp dequantize,
                      mlir::Value source, mlir::Value scale,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value bodyVL) const;

  /// widening_convert(%source,%vl){kind} lowers to ONE call, byte-identical to
  /// the legacy signed widening-conversion oracle:
  ///   result = __riscv_vwcvt_x_x_v_<resultDtype><resultLmul>(source, vl);
  /// The widened RESULT vector type (i32/m1 for i16mf2->i32m1, i64/m2 for
  /// i32m1->i64m2) drives the intrinsic suffix; the source is the loaded narrow
  /// vector. Only the two bounded SIGNED widening kinds the op verifier accepts
  /// (`sign_extend_widen_vf2` i16mf2->i32m1 and `widen_i32_to_i64` i32m1->i64m2)
  /// are convertible; both map to the same signed `vwcvt_x_x_v` callee. Any
  /// other kind or an unexpected source/result type pairing fails the match so
  /// the body falls back to the legacy validator unchanged (no mislower -- the
  /// unsigned widening convert would need `vwcvtu`, which this does NOT emit).
  mlir::LogicalResult
  emitWideningConvert(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, tcrvrvv::WideningConvertOp convert,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value bodyVL) const;

  /// masked_binary(%mask,%passthrough,%lhs,%rhs,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy masked merge sequence:
  ///   active = __riscv_v<op>_vv_<dtype><lmul>(lhs, rhs, vl);
  ///   result = __riscv_vmerge_vvm_<dtype><lmul>(passthrough, active, mask, vl);
  mlir::LogicalResult
  emitMaskedBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::MaskedBinaryOp masked,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const;

  /// macc(%lhs,%rhs,%accumulator,%vl){kind} ->
  ///   __riscv_vmacc_vv_<dtype><lmul>(accumulator, lhs, rhs, vl)
  /// The fused multiply-accumulate (acc += lhs * rhs) writes into the
  /// accumulator vector, so the C call order is (accumulator, lhs, rhs, vl) --
  /// byte-identical to the legacy plain/scalar-broadcast MAcc compute step
  /// (RVVEmitCRoutePlanning oracle: `vmacc_vv_i32m1(acc_vec, lhs_vec, rhs_vec,
  /// vl)`). The scalar-broadcast rung is the SAME op whose rhs is fed by a
  /// tcrv_rvv.splat (lowered by emitSplat); only the operand source differs, the
  /// macc lowering is identical.
  ///
  /// Malformed-body guard: the legacy macc derivation (deriveMAccIntrinsic) is
  /// SEW32-only and requires the explicit separate-accumulator + output-store
  /// layout contracts. A macc whose kind/layout, (dtype, lmul) config, or
  /// operand mapping is outside this bounded slice is NOT lowered here --
  /// notifyMatchFailure rolls the conversion back so the legacy validator still
  /// sees (and rejects/owns) it. Type-correctness is preserved: every operand is
  /// the same typed vector, and the result type is resolved before any emitc op
  /// is created so a non-beachhead config rolls back cleanly.
  mlir::LogicalResult
  emitMAcc(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::MAccOp macc,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value bodyVL) const;

  /// masked_macc(%mask,%lhs,%rhs,%accumulator,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy computed-mask macc sequence:
  ///   active = __riscv_vmacc_vv_<dtype><lmul>(accumulator, lhs, rhs, vl);
  ///   result = __riscv_vmerge_vvm_<dtype><lmul>(accumulator, active, mask, vl);
  /// The fused macc multiplies/accumulates on every lane; the merge then keeps
  /// the ACCUMULATOR vector on inactive lanes (the passthrough) and the macc
  /// result on active (mask-true) lanes -- the same passthrough = accumulator
  /// contract the legacy oracle emits.
  mlir::LogicalResult
  emitMaskedMAcc(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 tcrvrvv::MaskedMAccOp masked,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const;

  /// strided_load(%abi,%stride,%vl) ->
  ///   off = i * stride; ptr = base + off;
  ///   bytestride = (ptrdiff_t)stride * (ptrdiff_t)4;
  ///   __riscv_vlse<sew>_v_<dtype><lmul>(ptr, bytestride, vl)
  mlir::LogicalResult
  emitStridedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::StridedLoadOp load,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// strided_store(%abi,%val,%stride,%vl) ->
  ///   off = i * stride; ptr = base + off;
  ///   bytestride = (ptrdiff_t)stride * (ptrdiff_t)4;
  ///   __riscv_vsse<sew>_v_<dtype><lmul>(ptr, bytestride, val, vl)
  mlir::LogicalResult
  emitStridedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::StridedStoreOp store,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// Resolve the segment2 field-vector (sew, lmul, dtype) facts plus the tuple C
  /// type and per-field emitc vector type. Returns false (the caller fails the
  /// match) for any element/lmul outside the named grid so a non-i32 / non-{m1}
  /// segment2 body falls back unconverted rather than being mislowered.
  struct Segment2Facts {
    unsigned sew = 0;
    llvm::StringRef lmul;
    llvm::StringRef dtype;
    mlir::Type fieldVecType;
    mlir::Type tupleType;
  };
  bool resolveSegment2Facts(mlir::ConversionPatternRewriter &rewriter,
                            tcrvrvv::VectorType fieldType,
                            Segment2Facts &out) const;

  /// The interleaved segment2 base pointer: `base + (i * 2)`. The interleaved
  /// memory holds the two fields adjacent per element, so the segment base
  /// advances by 2*chunk -- byte-identical to the legacy segment2 oracle
  /// (`size_t off = i * 2; ptr = base + off`).
  mlir::Value emitSegment2InterleavedPointer(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value base, mlir::Value inductionVar) const;

  /// segment2_load(%src,%vl) -> field0, field1. The interleaved deinterleave
  /// load reads one segment2 tuple from the interleaved source, then the two
  /// tcrv_rvv.move ops extract the fields (emitSegment2FieldExtract via
  /// emitMove). Here we emit ONLY the tuple load and record (tuple, index) for
  /// each field result so the move-sourced vget can resolve it:
  ///   ptr = src + (i * 2);
  ///   vint<sew>m<lmul>x2_t tuple = __riscv_vlseg2e<sew>_v_<dtype><lmul>x2(ptr, vl)
  mlir::LogicalResult emitSegment2Load(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::Segment2LoadOp segLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
          &segmentFieldMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// segment2_store(%dst,%field0,%field1,%vl). The interleave store packs the
  /// two field vectors into one tuple, then stores it to the interleaved
  /// destination (TWO emitted steps, byte-identical to the legacy oracle):
  ///   vint<sew>m<lmul>x2_t tuple = __riscv_vcreate_v_<dtype><lmul>x2(f0, f1);
  ///   ptr = dst + (i * 2);
  ///   __riscv_vsseg2e<sew>_v_<dtype><lmul>x2(ptr, tuple, vl)
  mlir::LogicalResult
  emitSegment2Store(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, tcrvrvv::Segment2StoreOp segStore,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_segment2_load(%src,%mask,%pass0,%pass1,%vl) -> field0, field1. The
  /// computed-mask segment2 load packs the two old-destination passthroughs into
  /// a tuple, masked-loads the interleaved source into a result tuple, then
  /// extracts the two fields (FOUR emitted steps, byte-identical to the legacy
  /// computed-mask segment2 load oracle):
  ///   pass = __riscv_vcreate_v_<dtype><lmul>x2(old0, old1);
  ///   ptr = src + (i * 2);
  ///   tuple = __riscv_vlseg2e<sew>_v_<dtype><lmul>x2_tumu(mask, pass, ptr, vl);
  ///   field0 = __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, 0);
  ///   field1 = __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, 1)
  /// The mask MUST come from a compare in the same scope (the computed-mask
  /// family authority); refuse any other producer so a malformed body falls
  /// back to the legacy validator.
  mlir::LogicalResult emitMaskedSegment2Load(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedSegment2LoadOp segLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_segment2_store(%dst,%mask,%field0,%field1,%vl). The computed-mask
  /// segment2 store packs the two payload fields into a tuple, then
  /// masked-stores it to the interleaved destination (TWO emitted steps,
  /// byte-identical to the legacy computed-mask segment2 store oracle):
  ///   tuple = __riscv_vcreate_v_<dtype><lmul>x2(f0, f1);
  ///   ptr = dst + (i * 2);
  ///   __riscv_vsseg2e<sew>_v_<dtype><lmul>x2_m(mask, ptr, tuple, vl)
  /// The mask MUST come from a compare in the same scope; refuse otherwise.
  mlir::LogicalResult emitMaskedSegment2Store(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedSegment2StoreOp segStore,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// Emit one __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, idx) field
  /// extract, with the step provenance comment carrying the (op name, role) of
  /// the segment op that owns the extract.
  mlir::Value emitSegment2FieldExtract(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      llvm::StringRef opName, llvm::StringRef role, llvm::StringRef callee,
      mlir::Value tuple, mlir::Type fieldVecType, unsigned index) const;

  /// move{copy}(%src,%vl) -> passthrough OR segment2 field extract. Two shapes:
  ///
  ///  (a) base-memory movement family: the loaded vector is marked as the store
  ///      value with a no-op copy move (structural body authority). The copy
  ///      carries no compute, so it maps the result SSA value to the same emitc
  ///      Value -- the legacy oracle emits NO call for it.
  ///
  ///  (b) segment2 deinterleave family: the move's source is a tcrv_rvv.move-
  ///      observed segment2_load field result. The legacy oracle emits a
  ///      __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, idx) extract at the
  ///      move's position (role=compute). Detect the segmentFieldMap entry and
  ///      emit the vget; otherwise the plain identity copy.
  ///
  /// Only kind = "copy" is in this bounded slice; any other movement kind falls
  /// back so a semantically meaningful move is never silently dropped.
  mlir::LogicalResult
  emitMove(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::MoveOp move,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
               &segmentFieldMap) const;

  /// index_load(%abi,%vl) ->
  ///   ptr = index_buf + i; __riscv_vle<eew>_v_u<eew>m<lmul>(ptr, vl)
  /// Loads the UNSIGNED element-index/offset vector for an indexed gather/
  /// scatter. The index buffer is read unit-stride (`index_buf + i`), exactly
  /// like a plain load but into the unsigned index vector type -- byte-identical
  /// to the legacy index_load oracle (`__riscv_vle32_v_u32m1(index + i, vl)`).
  mlir::LogicalResult
  emitIndexLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                tcrvrvv::IndexLoadOp indexLoad,
                llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// The single masked indexed gather/scatter op that consumes `indexLoad`'s
  /// result, or null if the index feeds a plain (base-memory) indexed op. The
  /// computed-mask indexed path scales the index early (see emitIndexLoad); the
  /// plain path scales inside its own emitter.
  static mlir::Operation *maskedIndexedConsumer(tcrvrvv::IndexLoadOp indexLoad);

  /// The data (payload) vector type of a masked indexed gather (its loaded
  /// result) or scatter (its stored value), used to size the element->byte
  /// index scale.
  static tcrvrvv::VectorType
  maskedIndexedDataVectorType(mlir::Operation *maskedConsumer);

  /// indexed_load(%data,%indices,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   loaded = __riscv_vloxei<eew>_v_<dtype><lmul>(data_base, bytes, vl)
  /// The element index vector is scaled to a BYTE offset vector, then the
  /// ordered indexed (gather) access reads `data_base[byte_offset]` per lane.
  /// The data base is NOT offset by the induction var (a gather reads scattered
  /// elements relative to the buffer head) -- byte-identical to the legacy
  /// indexed_load oracle.
  mlir::LogicalResult
  emitIndexedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::IndexedLoadOp indexedLoad,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value bodyVL) const;

  /// indexed_store(%dst,%indices,%value,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   __riscv_vsoxei<eew>_v_<dtype><lmul>(dst_base, bytes, value, vl)
  /// The element index vector is byte-scaled, then the ordered indexed
  /// (scatter) access writes `dst_base[byte_offset] = value[lane]`. The dst
  /// base is NOT offset by the induction var -- byte-identical to the legacy
  /// indexed_store oracle. Only the unique-index slice is accepted (duplicate
  /// resolution is not modeled).
  mlir::LogicalResult
  emitIndexedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::IndexedStoreOp indexedStore,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const;

  /// The element->byte index scale shared by indexed load and store:
  ///   __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl)
  /// where elemBytes is the data element byte width (4 for i32). The verbatim
  /// step comment is carried from the indexed source op so the rendered C keeps
  /// the `callee=__riscv_vmul_vx_u32m1` provenance line.
  mlir::Value
  emitIndexByteScale(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, llvm::StringRef sourceOpName,
                     llvm::StringRef sourceRole, mlir::Value indices,
                     mlir::Type indexEmitCType,
                     tcrvrvv::IndexVectorType indexVecType,
                     tcrvrvv::VectorType dataVectorType,
                     mlir::Value bodyVL) const;

  /// mask_load(%abi,%vl) -> TWO calls:
  ///   maskvec = __riscv_vle<sew>_v_<dtype><lmul>(mask_buf + i, vl);
  ///   mask    = __riscv_vmsne_vx_<dtype><lmul>_b<maskbits>(maskvec, 0, vl)
  /// The base-memory masked families compute their predicate from a runtime mask
  /// BUFFER: load it unit-stride as a data vector, then test each lane != 0 to
  /// produce the vbool predicate -- byte-identical to the legacy mask_load
  /// oracle. The mask is genuine mask_load authority (NOT a compare on data),
  /// which is exactly the legality the negative fixtures require.
  mlir::LogicalResult
  emitMaskLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               tcrvrvv::MaskLoadOp maskLoad,
               llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
               mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_load(%abi,%mask,%passthrough,%vl) ->
  ///   ptr = src + i;
  ///   __riscv_vle<sew>_v_<dtype><lmul>_tumu(mask, passthrough, ptr, vl)
  /// The masked unit-stride load reads the source unit-stride but only writes
  /// active (mask-true) lanes; inactive/tail lanes keep the passthrough vector
  /// (the old destination) via the _tumu policy form -- byte-identical to the
  /// legacy masked_load oracle. The mask MUST come from mask_load authority (not
  /// a data compare): reject a compare-sourced mask so a masked body that lacks
  /// explicit mask_load authority falls back (the negative-fixture contract).
  mlir::LogicalResult
  emitMaskedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 tcrvrvv::MaskedLoadOp maskedLoad,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_store(%abi,%mask,%value,%vl) ->
  ///   ptr = dst + i;
  ///   __riscv_vse<sew>_v_<dtype><lmul>_m(mask, ptr, value, vl)
  /// The masked unit-stride store writes only active (mask-true) lanes; inactive
  /// and tail lanes keep their memory contents (no passthrough needed -- the
  /// store simply skips them) -- byte-identical to the legacy masked_store
  /// oracle. The mask MUST come from explicit mask_load authority (NOT a data
  /// compare): the negative fixture rejects a compare-sourced masked store.
  mlir::LogicalResult
  emitMaskedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::MaskedStoreOp maskedStore,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_strided_load(%abi,%mask,%passthrough,%stride,%vl) ->
  ///   ptr = (elem_t*)((uint8_t*)src + i * stride);
  ///   __riscv_vlse<sew>_v_<dtype><lmul>_tumu(mask, passthrough, ptr, stride, vl)
  /// The computed-mask masked byte-strided load reads the source at a runtime
  /// BYTE stride but only writes active (mask-true) lanes; inactive/tail lanes
  /// keep the passthrough (old-destination) vector via the _tumu policy form --
  /// byte-identical to the legacy computed-mask masked-strided-load oracle. The
  /// stride MUST be a byte-stride ABI role and the mask MUST be compare/mask_load
  /// authority; otherwise the malformed body falls back.
  mlir::LogicalResult emitMaskedStridedLoad(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedStridedLoadOp maskedLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_strided_store(%abi,%mask,%value,%stride,%vl) ->
  ///   ptr = (elem_t*)((uint8_t*)dst + i * stride);
  ///   __riscv_vsse<sew>_v_<dtype><lmul>_m(mask, ptr, stride, value, vl)
  /// The computed-mask masked byte-strided store writes only active (mask-true)
  /// lanes at a runtime BYTE stride; inactive/tail lanes keep their memory
  /// contents -- byte-identical to the legacy computed-mask masked-strided-store
  /// oracle. Byte-stride role + compare/mask_load mask authority required.
  mlir::LogicalResult emitMaskedStridedStore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedStridedStoreOp maskedStore,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const;

  /// masked_indexed_load(%data,%indices,%mask,%passthrough,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   loaded = __riscv_vluxei<eew>_v_<dtype><lmul>_tumu(mask, passthrough,
  ///                                                     data_base, bytes, vl)
  /// The computed-mask masked indexed gather byte-scales the element indices,
  /// then reads scattered elements but only writes active (mask-true) lanes
  /// (passthrough preserved on inactive/tail lanes via _tumu) -- byte-identical
  /// to the legacy computed-mask indexed-gather oracle. Only the element-offset,
  /// EEW=32 slice with compare/mask_load mask authority is accepted.
  mlir::LogicalResult emitMaskedIndexedLoad(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedIndexedLoadOp maskedLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// masked_indexed_store(%dst,%indices,%mask,%value,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   __riscv_vsoxei<eew>_v_<dtype><lmul>_m(mask, dst_base, bytes, value, vl)
  /// The computed-mask masked indexed scatter byte-scales the element indices,
  /// then writes active (mask-true) lanes to scattered byte offsets; inactive/
  /// tail lanes are skipped -- byte-identical to the legacy computed-mask
  /// indexed-scatter oracle. Only the element-offset, unique-index, EEW=32 slice
  /// with compare/mask_load mask authority is accepted.
  mlir::LogicalResult emitMaskedIndexedStore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedIndexedStoreOp maskedStore,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const;

  /// True iff `bufferValue` is an emitc pointer whose pointee names the unsigned
  /// 32-bit index element ("uint32_t"). The index buffer C type
  /// (e.g. "const uint32_t *") becomes the index pointer; the index_load reads
  /// it at u32 width.
  static bool indexBufferIsU32(mlir::Value bufferValue);

  /// True iff `bufferValue` is an emitc pointer whose pointee names the mask
  /// element scalar ("int32_t" / "int64_t"). The mask buffer is loaded at the
  /// data element width before the nonzero test.
  static bool maskBufferPointeeMatches(mlir::Value bufferValue,
                                       llvm::StringRef dtype);

  /// Convert a `!tcrv_rvv.index_vector<...>` to its EmitC type, accepting only a
  /// genuinely-lowered emitc type (see convertVectorTypeToEmitC for why the
  /// identity fallback must be rejected).
  mlir::Type
  convertIndexVectorTypeToEmitC(tcrvrvv::IndexVectorType type) const;

  /// Scaled element pointer: off = induction * stride; ptr = base + off.
  /// Mirrors the legacy materializer parseScaledPointerExpression path
  /// (`base + (induction * stride)`): an emitc.mul of the size_t induction and
  /// stride, then an emitc.add onto the pointer base.
  mlir::Value emitScaledPointer(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value base,
                                mlir::Value inductionVar,
                                mlir::Value stride) const;

  /// Byte stride: (ptrdiff_t)stride * (ptrdiff_t)4. Mirrors the legacy
  /// materializer parseSimpleProductExpression path for `stride * 4` with
  /// cType "ptrdiff_t": cast the runtime stride and the element-size literal to
  /// ptrdiff_t, then multiply.
  mlir::Value emitByteStride(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value stride,
                             tcrvrvv::VectorType vectorType) const;

  /// True when the strided op carries a runtime BYTE stride (the base-memory
  /// movement family) rather than an element stride (the elementwise family).
  /// The two are distinguished by the stride's defining runtime ABI value role:
  /// `source-byte-stride` / `destination-byte-stride` are byte strides passed
  /// AS-IS to vlse/vsse; `*-input-stride` / `output-stride` are element strides
  /// the elementwise path scales by the element width. Returning the wrong one
  /// would emit numerically wrong addressing, so the distinction is taken
  /// straight from the typed ABI role fact, not a heuristic.
  static bool isByteStride(mlir::Value strideToken);

  /// True when the strided-load result is consumed by a tcrv_rvv.move (the
  /// base-memory strided movement shape: strided_load -> move{copy} -> store).
  /// The elementwise strided family feeds its strided_load into a compute op
  /// (binary), never a move, so this cleanly separates the two rungs.
  static bool loadedFeedsMove(tcrvrvv::StridedLoadOp load);

  /// True when the strided-store value is produced by a tcrv_rvv.move (the
  /// base-memory unit-load -> move{copy} -> strided_store shape). The
  /// elementwise strided store's value comes from a compute op, never a move.
  static bool storedValueFromMove(tcrvrvv::StridedStoreOp store);

  /// True iff a masked-memory predicate `mask` is produced by an in-family mask
  /// authority: either an explicit tcrv_rvv.mask_load buffer (the base-memory
  /// masked family) or a tcrv_rvv.compare in the same VL scope (the
  /// computed-mask memory family). Both lower to the byte-identical masked-load
  /// `_tumu` / masked-store `_m` forms, so the converter accepts either; any
  /// other producer is malformed and must fall back.
  static bool isMaskFromMaskLoadOrCompare(mlir::Value mask);

  /// True iff a segment2 store's field operand binds the EXPECTED field role,
  /// resolved structurally from the field vector's defining tcrv_rvv.load buffer
  /// ABI role. The interleave family carries the two field input loads as the
  /// segment2_store field0/field1 operands; a body that swaps them
  /// (segment2_store %dst, %field1, %field0) binds the wrong field role and is a
  /// malformed body the legacy provider rejects ("segment2_store to consume
  /// matching field0 and field1 load results"). Resolving the binding from the
  /// typed load buffer role (not the ABI c_name) honors I5. When the field
  /// operand is NOT a plain load (e.g. a computed-mask update's add result), the
  /// caller does not apply this guard.
  static bool fieldVectorBindsLoadRole(mlir::Value fieldVector,
                                       llvm::StringRef expectedRole);

  /// Byte-stride scaled pointer: ptr = (elem_t*)((uint8_t*)base + i * stride).
  /// The base-memory strided family receives a runtime BYTE stride, so the
  /// element pointer is computed in BYTE space: cast the element base to
  /// `uint8_t*` (preserving const), add `i * stride` bytes, then cast back to
  /// the element pointer type -- byte-identical to the legacy base-memory
  /// strided oracle (`(const uint8_t*)base + i*stride; (const int32_t*)...`).
  mlir::Value emitByteStridedPointer(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value base,
                                     mlir::Value inductionVar,
                                     mlir::Value stride) const;

  static unsigned vectorElementWidth(tcrvrvv::VectorType type);

  /// Convert a `!tcrv_rvv.vector<...>` to its EmitC type, but ONLY accept a
  /// result the beachhead converter genuinely lowered (an `emitc` type). The
  /// driver registers an identity fallback conversion so unrelated IR is never
  /// illegalized; that identity would otherwise pass an unhandled vector type
  /// (e.g. lmul m2) straight through, letting a half-converted call_opaque keep
  /// a `!tcrv_rvv.vector<...>` result and silently corrupt the module. Rejecting
  /// any non-emitc result here makes a non-beachhead family fail the match and
  /// roll back cleanly, so the export seam falls back to the legacy path.
  mlir::Type convertVectorTypeToEmitC(tcrvrvv::VectorType type) const;

  /// The elementwise binary mnemonic. Integer kinds map to the v-prefixed forms
  /// (vadd/vsub/vmul); float vectors (f32/f64) map to the f-prefixed forms
  /// (vfadd/vfsub/vfmul) -- the floating-point arithmetic family the SEW=64
  /// double-precision coverage rung needs. `isFloat` keys the f-prefix the same
  /// way compareMnemonic does, so a single emitBinary path covers both.
  static std::optional<llvm::StringRef>
  binaryMnemonic(llvm::StringRef kind, bool isFloat);

  /// The reduction mnemonic for tcrv_rvv.reduce / tcrv_rvv.standalone_reduce,
  /// mirroring the legacy getRVVSelectedBodyReductionIntrinsic /
  /// getRVVSelectedBodyStandaloneReductionIntrinsic kind tables (add -> vredsum,
  /// min -> vredmin, max -> vredmax). Unknown kinds fail the match so the body
  /// falls back to the legacy validators unchanged.
  static std::optional<llvm::StringRef> reductionMnemonic(llvm::StringRef kind);

  /// The vector-vector compare predicate mnemonic, mirroring the legacy
  /// getRVVSelectedBody{,Float}CompareIntrinsicForPredicate tables for the
  /// vv-form predicates the compare-select bodies use. Integer:
  /// eq/slt/sle -> vmseq/vmslt/vmsle. Float (f-prefixed):
  /// eq/slt/sle -> vmfeq/vmflt/vmfle.
  static std::optional<llvm::StringRef> compareMnemonic(llvm::StringRef kind,
                                                        bool isFloat);

  /// The mask-composition mnemonic for tcrv_rvv.mask_and. The Stage-2 slice
  /// supports only kind = "and" (-> vmand), matching the op's verifier.
  static std::optional<llvm::StringRef> maskAndMnemonic(llvm::StringRef kind);
};

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_RVV_RVVTOEMITCINTERNAL_H
