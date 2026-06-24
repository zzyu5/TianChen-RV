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

struct AbiParam {
  tcrvrvv::RuntimeABIValueOp op;
  std::string cType;
  mlir::Type emitcType;
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

  /// The 16x1-REPACKED full-GEMM recognizer: a with_vl scope whose ONLY compute
  /// op is a single tcrv_rvv.repack_gemm_q4_0_q8_0. The op identity is the
  /// dispatch key; the emitter owns the structured block-as-lane expansion.
  static bool isRepackGemmQ4_0Q8_0Body(tcrvrvv::WithVLOp scope);

  /// The 16x1-REPACKED single-column GEMV (decode) recognizer: a with_vl scope
  /// whose ONLY compute op is a single tcrv_rvv.repack_gemv_q4_0_q8_0. The op
  /// identity is the dispatch key; the emitter owns the structured block-as-lane
  /// single-output-column expansion.
  static bool isRepackGemvQ4_0Q8_0Body(tcrvrvv::WithVLOp scope);

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

  /// The Family-A sibling recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q8_0_q8_0_block_dot.
  static bool isQ8_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The BINARY-class sibling recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.q1_0_q8_0_block_dot. The op identity is the dispatch
  /// key; the emitter owns the structured binary-sign-decode expansion.
  static bool isQ1_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The Family-B sibling recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q4_1_q8_1_block_dot.
  static bool isQ4_1Q8_1BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The Family-A 5-bit sibling recognizer: a with_vl scope whose ONLY compute
  /// op is a single tcrv_rvv.q5_0_q8_0_block_dot.
  static bool isQ5_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The Family-B 5-bit sibling recognizer: a with_vl scope whose ONLY compute
  /// op is a single tcrv_rvv.q5_1_q8_1_block_dot.
  static bool isQ5_1Q8_1BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The Family-A CODEBOOK sibling recognizer: a with_vl scope whose ONLY compute
  /// op is a single tcrv_rvv.iq4_nl_q8_0_block_dot.
  static bool isIQ4NLQ8_0BlockDotBody(tcrvrvv::WithVLOp scope);

  /// The CODEBOOK SUPER-BLOCK recognizer: a with_vl scope whose ONLY compute op
  /// is a single tcrv_rvv.iq4_xs_q8_k_block_dot (the super-block variant of
  /// iq4_nl: iq4_nl's codebook gather + the q4_K-style super-block signed scale).
  static bool isIQ4XSQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The GRID-codebook recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.iq2_xxs_q8_k_block_dot (the FIRST member of the deep IQ tail:
  /// the packed uint64 grid codebook + sign plane + integer-domain 4-bit scale).
  static bool isIQ2XXSQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The GRID-codebook sibling recognizer: a with_vl scope whose ONLY compute op is
  /// a single tcrv_rvv.iq2_xs_q8_k_block_dot (the SECOND member of the deep IQ tail:
  /// the 512-entry packed uint64 grid codebook + 9-bit-index uint16 qs + sign plane +
  /// explicit per-sub-block 4-bit scales[]).
  static bool isIQ2XSQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The GRID-codebook sibling recognizer: a with_vl scope whose ONLY compute op is
  /// a single tcrv_rvv.iq2_s_q8_k_block_dot (the THIRD member of the deep IQ tail:
  /// the 1024-entry packed uint64 grid codebook + 10-bit index from a single qs byte
  /// plus 2 qh-plane bits + EXPLICIT signs read directly from a sign-byte region +
  /// explicit per-sub-block 4-bit scales[]).
  static bool isIQ2SQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The GRID-codebook sibling recognizer: a with_vl scope whose ONLY compute op is
  /// a single tcrv_rvv.iq3_xxs_q8_k_block_dot (a member of the deep IQ tail: the
  /// 256-entry packed uint32 grid-of-4 codebook + ksigns_iq2xs sign plane + a packed
  /// per-sub-block aux uint32 carrying the sign selectors + the 4-bit scale).
  static bool isIQ3XXSQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The GRID-codebook sibling recognizer: a with_vl scope whose ONLY compute op is
  /// a single tcrv_rvv.iq3_s_q8_k_block_dot (a member of the deep IQ tail; a
  /// re-composition of the 512-entry grid-of-4 (iq3_xxs), the qh 9th-bit plane
  /// (iq2_s), the EXPLICIT signs read from memory (iq2_s), and the explicit
  /// per-sub-block 4-bit scales (iq2_s)).
  static bool isIQ3SQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The TERNARY-grid recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.iq1_s_q8_k_block_dot (the TERNARY class -- the last common ggml dot
  /// kernel: the 2048-entry ternary grid byte-viewed as signed int8, the uint16
  /// qh-encoded 3-bit scale, the 11-bit qs+qh index, and the per-block DELTA term
  /// folded via the q8 bsums).
  static bool isIQ1SQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The TERNARY-grid IQ1_M recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.iq1_m_q8_k_block_dot (the LAST common ggml dot kernel -- the
  /// iq1_s sibling: the SAME 2048-entry ternary grid, the packed-iq1m_scale fp16
  /// reconstruction from the 4 scales[] words, the uint8 qh[16] plane, and the
  /// per-group DELTA term built from a parallel Σq8 reduction).
  static bool isIQ1MQ8KBlockDotBody(tcrvrvv::WithVLOp scope);

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

  /// The K-quant K2 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q6_k_q8_k_block_dot (the Q6_K x Q8_K super-block FULL
  /// block dot-product producing the fp32 *s).
  static bool isQ6_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The q4_K K4a recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q4_k_q8_k_aux_partial (the Q4_K x Q8_K super-block integer aux32 +
  /// decoded scale/min partial -- the INTEGER CORE before the fp32 d/dmin fold).
  static bool isQ4_KQ8_KAux32PartialBody(tcrvrvv::WithVLOp scope);

  /// The q4_K K4b recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q4_k_q8_k_block_dot (the Q4_K x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- K4a's integer core + the deferred fp32
  /// fold + the q4_K min term).
  static bool isQ4_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The q5_K recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q5_k_q8_k_block_dot (the Q5_K x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- q4_K's integer core + the qh 5th-bit
  /// injection + the deferred fp32 fold + the q4_K min term).
  static bool isQ5_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The q2_K recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q2_k_q8_k_block_dot (the Q2_K x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- the 2-bit weight unpack + the 4-bit
  /// nibble scale/min extraction + the SCALAR fp32 fold + the q2_K min term).
  static bool isQ2_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope);

  /// The q3_K recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q3_k_q8_k_block_dot (the Q3_K x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- q2_K's 2-bit qs unpack + the
  /// SUBTRACTIVE hmask high-bit injection (q5_K's plane, -4 when unset -> signed
  /// [-4,3]) + the q3_K SIGNED 6-bit scale dance + q6_K's NO-min deferred fp32
  /// fold).
  static bool isQ3_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope);

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

  /// The standalone reduction intrinsic name:
  ///   __riscv_v<red>_vs_<srcDtype><srcLmul>_<resultDtype>m1
  /// For the non-widening forms srcDtype==resultDtype and srcLmul is the input
  /// lmul (e.g. vredsum_vs_i32m2_i32m1). For the widening forms the source is a
  /// narrower fractional-lmul vector (i16/mf2) and the result is i32/m1
  /// (vwredsum_vs_i16mf2_i32m1). Byte-identical to the legacy
  /// getRVVSelectedBodyStandaloneReductionIntrinsic shape.
  static std::string standaloneReductionIntrinsicName(
      llvm::StringRef mnemonic, llvm::StringRef srcDtype,
      llvm::StringRef srcLmul, llvm::StringRef resultDtype);

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

  /// Emit the COMPLETE ggml ggml_vec_dot_q5_0_q8_0 block dot-product for one
  /// tcrv_rvv.q5_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-A 5-bit sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop /
  /// unroll / tail / scale-read / store / strip scaffolding STRUCTURE; the two
  /// kernel-specific differences are (a) the integer core decodes a 5-BIT weight
  /// (the q4_1 unsigned nibble unpack PLUS the per-element 5th high bit from the
  /// 32-bit qh field PLUS the offset-binary `-16` bias, via
  /// emitFiveBitOffsetBinaryDecodeProductValue) and (b) the fp32 fold is ggml's
  /// q5_0 order `sumf = sumf + (d_x*d_y)*sumi` (the dual scales multiplied FIRST,
  /// q8_0's order -- NOT the Q4_0 sibling's `((sumi*d_x)*d_y)`).
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*22;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);   // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     uint16_t qhLo = *(const uint16_t *)(xb + 2);  // 5th-bit field, ALIGNED
  ///     uint16_t qhHi = *(const uint16_t *)(xb + 4);  // (two halves, not u32)
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 16; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
  ///       vuint8<L>_t w  = __riscv_vle8_v_u8<L>(xb + 6 + c, vl);
  ///       vint8<L>_t  y0 = __riscv_vle8_v_i8<L>(yb + 2 + c, vl);
  ///       vint8<L>_t  y1 = __riscv_vle8_v_i8<L>(yb + 2 + 16 + c, vl);
  ///       vint16<W>_t p  = <5-bit offset-binary decode/product>(w, y0, y1,
  ///                          qh&0xFFFF, qh>>16, c, vl);
  ///       vint32m1_t  seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t  red  = __riscv_vwredsum_vs_i16<W>_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + (d_x * d_y) * (float)sumi;   // ggml q5_0 order
  ///   }
  ///   *s = sumf;
  /// The strip loop anchors at integer_core_lmul ("mf4" default / "m1") exactly as
  /// the Q4_0 sibling; the nibble half-block shape is byte-identical. The 5th-bit
  /// injection uses a `vid + c` per-lane shift so it is correct for any chunk
  /// offset c (the mf4 multi-strip path) and any VLEN. The block-format facts are
  /// the op's typed attrs (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ5_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_q5_1_q8_1 block dot-product for one
  /// tcrv_rvv.q5_1_q8_1_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-B 5-bit sibling that COMBINES the two prior breadth increments,
  /// sharing the block-loop / unroll / tail / strip scaffolding STRUCTURE; the
  /// two kernel-specific facts are (a) the integer core decodes an UNSIGNED 5-bit
  /// weight in [0,31] -- the SAME nibble unpack + 5th-bit injection q5_0 uses
  /// (emitFiveBitOffsetBinaryDecodeProductValue) but with applyOffsetBias=false
  /// so the `-16` bias is dropped (the bias lives in q5_1's separate per-block
  /// MIN scale, exactly like q4_1), and (b) the fp32 fold reads FOUR per-block
  /// fp16 scales (d_x,m_x at weight+0/+2; d_y,s_y at activation+0/+2) and folds
  /// ggml's EXACT statement sumf += (d_x*d_y)*sumi + m_x*s_y (IDENTICAL to q4_1).
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*24;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*36;
  ///     float d_x = (float)*(const _Float16 *)(xb);    // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     float m_x = (float)*(const _Float16 *)(xb+2);
  ///     float s_y = (float)*(const _Float16 *)(yb+2);
  ///     uint16_t qhLo = *(const uint16_t *)(xb + 4);   // 5th-bit field, ALIGNED
  ///     uint16_t qhHi = *(const uint16_t *)(xb + 6);   // (two halves, not u32)
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 16; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
  ///       vuint8<L>_t w  = __riscv_vle8_v_u8<L>(xb + 8 + c, vl);
  ///       vint8<L>_t  y0 = __riscv_vle8_v_i8<L>(yb + 4 + c, vl);
  ///       vint8<L>_t  y1 = __riscv_vle8_v_i8<L>(yb + 4 + 16 + c, vl);
  ///       vint16<W>_t p  = <5-bit UNSIGNED decode/product>(w, y0, y1,
  ///                          qh&0xFFFF, qh>>16, c, vl);   // NO -16 bias
  ///       vint32m1_t  seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t  red  = __riscv_vwredsum_vs_i16<W>_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + ((d_x * d_y) * (float)sumi + m_x * s_y);  // ggml q5_1
  ///   }
  ///   *s = sumf;
  /// The block-format facts (QK/strides/offsets/scale model) are the op's typed
  /// attrs (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ5_1Q8_1BlockDot(
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

  /// Emit the COMPLETE ggml q4_0 16x1-REPACKED GEMM hot kernel (the validated
  /// vlen128-q4_0-16x1 ggml_gemm_q4_0_16x1_q8_0 path) for one
  /// tcrv_rvv.repack_gemm_q4_0_q8_0 op as fully STRUCTURED emitc nodes (I5; ZERO
  /// raw() strings -- every value is a node in the IR graph). It is the
  /// BLOCK-AS-LANE sibling of emitQ4_0Q8_0Gemm: the repacked block_q4_0x16
  /// weight lays the 16 interleaved rows of a group across 16 vector lanes, so
  /// the dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum wall). At
  /// VLEN=128 an e16m1 vector holds 8 i16 lanes, so each 16-block group is
  /// processed as two disjoint contiguous halves of 8 lanes. The repacked
  /// nibbles already carry the offset-binary ^0x88 bias, so the decode is a
  /// plain vsll/vsra sign-extension (NO in-kernel vxor). The shape mirrors the
  /// validated kernel exactly (no tail; ragged edges go to _generic at
  /// dispatch):
  ///   for (size_t y = 0; y < nr/4; ++y) {
  ///     const block_q8_0x4 *a = (vy) + y*nb;
  ///     for (size_t x = 0; x < nc/16; ++x) {
  ///       const block_q4_0x16 *b = (vx) + x*nb;
  ///       for (size_t half = 0; half < 2; ++half) {  // rows 0..7, 8..15
  ///         size_t roff = half*8;
  ///         vfloat32m2_t sumf_{0..3} = vfmv_v_f(0,8);
  ///         for (size_t l = 0; l < nb; ++l) {
  ///           vint16m1_t sumi_{0..3}_{lo,hi} = vmv_v_x(0,8);
  ///           for (size_t i = 0; i < 16; ++i) {
  ///             b_packed = vle8(&b[l].qs[i*16+roff], 8);
  ///             b_lo = vsra(vsll(b_packed,4),4);  b_hi = vsra(b_packed,4);
  ///             sumi_c_lo = vwmacc_vx(sumi_c_lo, a[l].qs[i*4+c],    b_lo, 8);
  ///             sumi_c_hi = vwmacc_vx(sumi_c_hi, a[l].qs[64+i*4+c], b_hi, 8);
  ///           }
  ///           sumi_c = vwadd_vv(sumi_c_lo, sumi_c_hi, 8);
  ///           b_d = vle16(&b[l].d[roff], 8);
  ///           d_c = vfwmul_vf(b_d, *(_Float16*)&a[l].d[c], 8);
  ///           sumf_c = vfmacc_vv(sumf_c, vfcvt_f_x_v(sumi_c,8), d_c, 8);
  ///         }
  ///         vse32(s + (y*4+c)*bs + x*16 + roff, sumf_c, 8);
  ///       }
  ///     }
  ///   }
  /// Each output is byte-exact vs ggml_gemm_q4_0_16x1_q8_0_generic (the patch is
  /// already byte-exact vs _generic; structural fidelity to its node sequence is
  /// the byte-exactness proof). All intrinsics are emitc.call_opaque nodes; the
  /// scale fold feeds the raw _Float16 a[l].d[c] into vfwmul_vf (NO float cast,
  /// unlike the per-block dot's fp16Read).
  mlir::LogicalResult emitRepackGemmQ4_0Q8_0(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml q4_0 16x1-REPACKED GEMV (decode) hot kernel (the
  /// validated vlen128-q4_0-16x1 ggml_gemv_q4_0_16x1_q8_0 path) for one
  /// tcrv_rvv.repack_gemv_q4_0_q8_0 op as fully STRUCTURED emitc nodes (I5; ZERO
  /// raw() strings -- every value is a node in the IR graph). It is the
  /// SINGLE-OUTPUT-COLUMN (decode) sibling of emitRepackGemmQ4_0Q8_0: the weight
  /// side is the SAME block_q4_0x16 block-as-lane layout (the 16 interleaved
  /// rows of a group across 16 vector lanes, the dot accumulates LANE-WISE via
  /// vwmacc, NO cross-lane vredsum wall), but the activation is ONE plain
  /// block_q8_0 stream (stride 34, quants at +2) -- a single activation column.
  /// At VLEN=128 an e16m1 vector holds 8 i16 lanes, so each 16-block group is
  /// processed as two disjoint contiguous halves of 8 lanes INLINE (no half
  /// loop). The repacked nibbles already carry the offset-binary ^0x88 bias, so
  /// the decode is a plain vsll/vsra sign-extension (NO in-kernel vxor). The
  /// shape mirrors the validated kernel exactly (no tail; ragged edges go to
  /// _generic at dispatch):
  ///   const block_q8_0 *a = (vy);  // set once, indexed a[l], reused per x
  ///   for (size_t x = 0; x < nc/16; ++x) {
  ///     const block_q4_0x16 *b = (vx) + x*nb;
  ///     vfloat32m2_t sumf_a = vfmv_v_f(0,8);  // rows 0..7
  ///     vfloat32m2_t sumf_b = vfmv_v_f(0,8);  // rows 8..15
  ///     for (size_t l = 0; l < nb; ++l) {
  ///       vint16m1_t sumi_{a,b}_{lo,hi} = vmv_v_x(0,8);
  ///       for (size_t i = 0; i < 16; ++i) {
  ///         b_a = vle8(&b[l].qs[i*16+0], 8);  b_b = vle8(&b[l].qs[i*16+8], 8);
  ///         b_a_lo = vsra(vsll(b_a,4),4);  b_a_hi = vsra(b_a,4);  // and b_b
  ///         sumi_a_lo = vwmacc_vx(sumi_a_lo, a[l].qs[i],    b_a_lo, 8);
  ///         sumi_a_hi = vwmacc_vx(sumi_a_hi, a[l].qs[16+i], b_a_hi, 8);
  ///         sumi_b_lo = vwmacc_vx(sumi_b_lo, a[l].qs[i],    b_b_lo, 8);
  ///         sumi_b_hi = vwmacc_vx(sumi_b_hi, a[l].qs[16+i], b_b_hi, 8);
  ///       }
  ///       sumi_a = vwadd_vv(sumi_a_lo, sumi_a_hi, 8);  // and sumi_b
  ///       b_d_a = vle16(&b[l].d[0], 8);  b_d_b = vle16(&b[l].d[8], 8);
  ///       d_a = vfwmul_vf(b_d_a, *(_Float16*)&a[l].d, 8);  // and d_b
  ///       sumf_a = vfmacc_vv(sumf_a, vfcvt_f_x_v(sumi_a,8), d_a, 8);  // and _b
  ///     }
  ///     vse32(s + x*16 + 0, sumf_a, 8);  vse32(s + x*16 + 8, sumf_b, 8);
  ///   }
  /// Each output is byte-exact vs ggml_gemv_q4_0_16x1_q8_0_generic (the patch is
  /// already byte-exact vs _generic; structural fidelity to its node sequence is
  /// the byte-exactness proof). All intrinsics are emitc.call_opaque nodes; the
  /// scale fold feeds the raw _Float16 a[l].d into vfwmul_vf (NO float cast).
  mlir::LogicalResult emitRepackGemvQ4_0Q8_0(
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

  /// Emit the COMPLETE ggml ggml_vec_dot_q4_1_q8_1 block kernel for one
  /// tcrv_rvv.q4_1_q8_1_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-B sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop / unroll /
  /// tail / strip-elision scaffolding, differing in exactly two kernel-specific
  /// parts (the q4_1 quantization is scale+MIN, asymmetric):
  ///   (a) the integer core decodes UNSIGNED nibbles [0,15]
  ///       (emitUnsignedNibbleDecodeProductValue: vand 0x0F / vsrl 0x04 on the u8
  ///       weight lane -> reinterpret -> the SAME signed vwmul/vwmacc against the
  ///       plain q8 halves), NOT the offset-binary `-8` chain. The weight chunk is
  ///       therefore loaded as u8 (the activations stay i8);
  ///   (b) the fold reads FOUR per-block fp16 scales (d_x,m_x at weight+0/+2;
  ///       d_y,s_y at activation+0/+2) and folds ggml's EXACT statement
  ///       sumf += (d_x*d_y)*sumi + m_x*s_y -- the (d_x*d_y) scale product times
  ///       sumi and the m_x*s_y MIN/SUM correction are SUMMED FIRST (the `+`
  ///       binds before the `+=`), then added to sumf: sumf + (A + B), an
  ///       fp-significant grouping required for byte-exactness.
  /// The block-format facts (QK/strides/offsets/scale model) are the op's typed
  /// attrs (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ4_1Q8_1BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_q8_0_q8_0 block kernel for one
  /// tcrv_rvv.q8_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-A sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop / unroll /
  /// tail / scale-read / store scaffolding STRUCTURE; the ONLY kernel-specific
  /// differences are (a) the integer core is a plain 32-element signed widening
  /// product (vle8 x2 -> vwmul_vv i8->i16 -> vwredsum, NO nibble decode / offset-
  /// binary bias / high-low split) and (b) the fp32 fold is ggml's q8_0 order
  /// `sumf = sumf + (float)sumi * (d_x * d_y)` (scales multiplied FIRST, distinct
  /// from the Q4_0 sibling's `((sumi*d_x)*d_y)`).
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*34;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);   // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 32; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e8m2(32 - c);     // m2: whole block @128
  ///       vint8m2_t  vx0 = __riscv_vle8_v_i8m2(xb + 2 + c, vl);
  ///       vint8m2_t  vy0 = __riscv_vle8_v_i8m2(yb + 2 + c, vl);
  ///       vint16m4_t p   = __riscv_vwmul_vv_i16m4(vx0, vy0, vl);
  ///       vint32m1_t seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t red  = __riscv_vwredsum_vs_i16m4_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + (float)sumi * (d_x * d_y);   // ggml q8_0 order
  ///   }
  ///   *s = sumf;
  /// The strip loop anchors at integer_core_lmul: "m2" (the whole 32-element
  /// block in one strip + one vwredsum at VLEN=128, matching ggml's hand-written
  /// reduction anchor), "m1" (two 16-element strips), or "mf4" (eight 4-element
  /// strips). All are byte-exact and VLEN-robust via the sumi-carrying seed. The
  /// elided shape (single vsetvl_e8m2(32), no inner loop) is correct only at
  /// VLEN >= 128 and requires the m2 anchor (verifier-enforced).
  mlir::LogicalResult emitQ8_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq4_nl_q8_0 block dot-product for one
  /// tcrv_rvv.iq4_nl_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-A CODEBOOK sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop /
  /// unroll / tail / scale-read / store scaffolding STRUCTURE and its block byte
  /// shape (stride 18, nibbles at +2, the low/high nibble <-> q8[0..15]/q8[16..31]
  /// split). The TWO kernel-specific facts are (a) the integer core's weight
  /// decode is a 16-entry NON-LINEAR codebook GATHER -- the 4-bit nibble indexes
  /// kvalues_iq4nl[16] via vrgather (ggml's exact RVV method) instead of the
  /// q4_0 offset-binary `nibble - 8` arithmetic -- and (b) the fp32 fold is ggml's
  /// iq4_nl order `sumf = sumf + (float)sumi * (d_x * d_y)` (scales-first, the
  /// q8_0/q5_0 order; ggml's _generic is `d * (sumi1 + sumi2)` with `d = d_y*d_x`).
  ///
  ///   static const int8_t tcrv_iq4_nl_kvalues[16] = { ... };  // codebook decl
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   vint8m1_t values = __riscv_vle8_v_i8m1(tcrv_iq4_nl_kvalues, 16); // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;            // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     int32_t sumi = 0;
  ///     size_t vl = __riscv_vsetvl_e8m1(16);       // m1: whole half-block @>=128
  ///     vuint8m1_t w   = __riscv_vle8_v_u8m1(xb + 2, vl);     // packed nibbles
  ///     vint8m1_t  y0  = __riscv_vle8_v_i8m1(yb + 2, vl);     // q8[0..15]
  ///     vint8m1_t  y1  = __riscv_vle8_v_i8m1(yb + 18, vl);    // q8[16..31]
  ///     vuint8m1_t iL  = __riscv_vand_vx_u8m1(w, 0x0F, vl);   // low nibble idx
  ///     vuint8m1_t iH  = __riscv_vsrl_vx_u8m1(w, 0x04, vl);   // high nibble idx
  ///     vint8m1_t  v0  = __riscv_vrgather_vv_i8m1(values, iL, vl);  // codebook!
  ///     vint8m1_t  v1  = __riscv_vrgather_vv_i8m1(values, iH, vl);
  ///     vint16m2_t p   = __riscv_vwmul_vv_i16m2(v0, y0, vl);  // low product
  ///     p              = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl); // + high
  ///     vint32m1_t red = __riscv_vwredsum_vs_i16m2_i32m1(p, seed, vl);
  ///     sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     sumf = sumf + (float)sumi * (d_x * d_y);    // ggml iq4_nl order
  ///   }
  ///   *s = sumf;
  /// The codebook GATHER pins the m1 anchor: vrgather's source/index/dest share one
  /// vtype, so to index all 16 table entries the `values` register's VLMAX must be
  /// >= 16; at VLEN=128 e8 m1 -> VLMAX=16 (the half-block), mf4 -> 4 (drops 12 of
  /// 16). The verifier enforces m1 (the codebook class is inherently Zvl128b-gated).
  /// The single-vsetvl_e8m1(16) half-block cover is byte-exact for ANY VLEN >= 128.
  /// The "robust" strip form keeps an inner strip loop carrying sumi (so a VLEN<128
  /// board would re-strip), but at m1 the half-block is one strip at VLEN>=128.
  mlir::LogicalResult emitIQ4NLQ8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

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

  /// Emit the COMPLETE ggml ggml_vec_dot_iq2_xxs_q8_K super-block dot-product for
  /// one tcrv_rvv.iq2_xxs_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// raw()). It is the FIRST member of the deep IQ tail -- the GRID-codebook class.
  /// Unlike the 16-entry per-nibble gather (iq4_nl/iq4_xs), each weight byte INDEXES
  /// a 256-entry packed uint64 GRID codebook (iq2xxs_grid), the per-element sign
  /// comes from a separate SIGN PLANE (ksigns_iq2xs/kmask), and the 4-bit scale folds
  /// in the INTEGER domain (q6_K-style bsum), mirroring _generic (quants.c:855-895):
  ///   static const int64_t tcrv_iq2xxs_grid[256]   = { ... };  // GRID codebook
  ///   static const uint8_t tcrv_iq2xxs_ksigns[128] = { ... };  // SIGN plane
  ///   static const uint8_t tcrv_iq2xxs_kmask[8] = {1,2,4,8,16,32,64,128};
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_iq2xxs_kmask, 8);   // ONCE
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq2xxs_grid;
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*66;  const uint8_t *yb = vy + ibl*292;
  ///     float d = (float)*(const _Float16 *)(xb) * *(const float *)(yb);
  ///     const uint8_t *qs = xb + 2;  const int8_t *q8 = yb + 4;
  ///     int32_t bsum = 0;
  ///     for (size_t ib32 = 0; ib32 < 8; ++ib32) {        // FLAT unrolled
  ///       // aux0/aux1 reassembled from 4 little-endian byte loads (2-aligned qs):
  ///       const uint8_t *a = qs + ib32*8;
  ///       int aux1 = a[4] | a[5]<<8 | a[6]<<16 | a[7]<<24;
  ///       int ls = 2*(aux1 >> 28) + 1;
  ///       int32_t sumi = 0;  (chained i32m1 reduction seed)
  ///       for (l=0..3) {  // each group of 8 elements
  ///         int idx   = a[l];
  ///         int signs = tcrv_iq2xxs_ksigns[(aux1 >> 7*l) & 127];
  ///         vsetvl_e8m1(8); grid = vle8_i8m1(grid_i8 + idx*8); q8v = vle8_i8m1(q8);
  ///         m = vmsne(vand(vmv(signs), kmask), 0);          // sign-bit mask
  ///         g = vmerge(grid, vneg(grid), m);                // apply signs
  ///         p = vwmul_i16m2(g, q8v);  sumi = vwredsum(p, sumi);  q8 += 8;
  ///       }
  ///       bsum += sumi * ls;
  ///     }
  ///     sumf = sumf + d * (float)bsum;                     // ONE emitc.expression
  ///   }
  ///   *s = 0.125f * sumf;                                  // SEPARATE statement
  /// The grid lookup is an indexed vle8(8) over a pointer (grid_i8 + idx*8), the
  /// operative vector width is 8 (one group), so there is NO vrgather table-index
  /// legality fact (no m1 table anchor). All vector ops use the m1 anchor (8 lanes
  /// fit well within VLMAX=16 at e8m1). The integer chained vwredsum reduction is
  /// order-free (integer add is associative); the per-super-block fp32 fold is
  /// invoked in STRICT ascending order so fp non-associativity is byte-exact across
  /// all -ffp-contract modes; the trailing 0.125f multiply stays OUT of the
  /// accumulate expression.
  mlir::LogicalResult emitIQ2XXSQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq3_xxs_q8_K super-block dot-product for one
  /// tcrv_rvv.iq3_xxs_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no raw()).
  /// It is a member of the deep IQ tail and the SIBLING of emitIQ2XXSQ8KBlockDot -- it
  /// REUSES the iq2_xxs grid + sign mechanism (the indexed packed GRID lookup, the
  /// ksigns_iq2xs/kmask SIGN plane, the broadcast/vand/vmsne/vneg/vmerge per-lane sign
  /// apply, the signed widening product + chained vwredsum, the packed-aux 4-bit scale
  /// 2*(aux>>28)+1 + the (aux>>7l)&127 sign selectors, the integer bsum fold, the
  /// per-super-block fp32 fold) with ONE load-bearing structural delta vs iq2_xxs,
  /// mirroring _generic (quants.c:999-1041):
  ///   static const uint32_t tcrv_iq3xxs_grid[256]  = { ... };  // GRID-of-4 codebook
  ///   static const uint8_t  tcrv_iq3xxs_ksigns[128] = { ... }; // SIGN plane (reused)
  ///   static const uint8_t  tcrv_iq3xxs_kmask[8] = {1,2,4,8,16,32,64,128};
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   vuint8m1_t kmaskLo = vle8(kmask,   4);  // {1,2,4,8}    for grid1 lanes 0..3
  ///   vuint8m1_t kmaskHi = vle8(kmask+4, 4);  // {16,32,64,128} for grid2 lanes 4..7
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq3xxs_grid;
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*98;  const uint8_t *yb = vy + ibl*292;
  ///     float d = (float)*(const _Float16 *)(xb) * *(const float *)(yb);
  ///     const uint8_t *q3 = xb + 2;          // 64 grid index bytes
  ///     const uint8_t *gas = xb + 66;        // 32 aux bytes (one uint32 / sub-block)
  ///     const int8_t  *q8 = yb + 4;          int32_t bsum = 0;
  ///     for (size_t ib32 = 0; ib32 < 8; ++ib32) {        // FLAT unrolled
  ///       // aux32 reassembled from 4 little-endian byte loads (2-aligned gas):
  ///       const uint8_t *a = gas + ib32*4;
  ///       uint32 aux32 = a[0] | a[1]<<8 | a[2]<<16 | a[3]<<24;
  ///       int ls = 2*(aux32 >> 28) + 1;
  ///       int32_t sumi = 0;  (chained i32m1 reduction seed)
  ///       const uint8_t *qg = q3 + ib32*8;   // 8 grid indices for this sub-block
  ///       for (l=0..3) {  // each group of 8 elements = TWO grid-of-4 indices
  ///         int signs = tcrv_iq3xxs_ksigns[(aux32 >> 7*l) & 127];
  ///         // pass A: grid1 = grid_i8 + qg[2l+0]*4 vs q8[0..3], kmask lanes 0..3
  ///         vsetvl_e8m1(4); g1 = vle8(grid_i8 + qg[2l+0]*4); a1 = vle8(q8+0);
  ///         m=vmsne(vand(vmv(signs),kmaskLo),0); g=vmerge(g1,vneg(g1),m);
  ///         p=vwmul_i16m2(g,a1); sumi=vwredsum(p,sumi);
  ///         // pass B: grid2 = grid_i8 + qg[2l+1]*4 vs q8[4..7], kmask lanes 4..7
  ///         vsetvl_e8m1(4); g2 = vle8(grid_i8 + qg[2l+1]*4); a2 = vle8(q8+4);
  ///         m=vmsne(vand(vmv(signs),kmaskHi),0); g=vmerge(g2,vneg(g2),m);
  ///         p=vwmul_i16m2(g,a2); sumi=vwredsum(p,sumi);  q8 += 8;
  ///       }
  ///       bsum += sumi * ls;
  ///     }
  ///     sumf = sumf + d * (float)bsum;                     // ONE emitc.expression
  ///   }
  ///   *s = 0.25f * sumf;                                   // SEPARATE statement
  /// The grid lookup is an indexed vle8(4) over a pointer (grid_i8 + idx*4); the
  /// operative width is 4 (one grid-of-4 entry), so there is NO vrgather table-index
  /// legality fact (no m1 table anchor). The integer chained vwredsum reduction is
  /// order-free; the per-super-block fp32 fold is invoked in STRICT ascending order so
  /// fp non-associativity is byte-exact across all -ffp-contract modes; the trailing
  /// 0.25f multiply stays OUT of the accumulate expression.
  mlir::LogicalResult emitIQ3XXSQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

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

  /// Emit the COMPLETE ggml ggml_vec_dot_iq2_xs_q8_K super-block dot-product for one
  /// tcrv_rvv.iq2_xs_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no raw()).
  /// It is the SECOND member of the deep IQ tail and the SIBLING of
  /// emitIQ2XXSQ8KBlockDot -- it REUSES the iq2_xxs grid + sign mechanism (indexed
  /// packed-uint64 GRID lookup, the ksigns_iq2xs/kmask SIGN plane, the
  /// broadcast/vand/vmsne/vneg/vmerge per-lane sign apply, the signed widening product
  /// + chained vwredsum, the integer bsum fold, the per-super-block fp32 fold, the
  /// 0.125f factor) with THREE structural deltas vs iq2_xxs, mirroring _generic
  /// (quants.c:897-945):
  ///   static const int64_t tcrv_iq2xs_grid[512]   = { ... };  // 512-entry GRID
  ///   static const uint8_t tcrv_iq2xs_ksigns[128] = { ... };  // SAME SIGN plane
  ///   static const uint8_t tcrv_iq2xs_kmask[8] = {1,2,4,8,16,32,64,128};
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_iq2xs_kmask, 8);   // ONCE
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq2xs_grid;
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*74;  const uint8_t *yb = vy + ibl*292;
  ///     float d = (float)*(const _Float16 *)(xb) * *(const float *)(yb);
  ///     const uint8_t *qs = xb + 2;  const uint8_t *sc = xb + 66;  const int8_t *q8 = yb + 4;
  ///     int32_t bsum = 0;
  ///     for (size_t ib32 = 0; ib32 < 8; ++ib32) {        // FLAT unrolled
  ///       int ls1 = 2*(sc[ib32] & 0xf) + 1;  int ls2 = 2*(sc[ib32] >> 4) + 1;
  ///       // half h=0: groups l=0,1 with ls1; half h=1: groups l=2,3 with ls2.
  ///       for (h = 0..1) {
  ///         int32_t sumi = 0;  (chained i32m1 reduction seed)
  ///         for (l = 2*h .. 2*h+1) {  // each group of 8 elements
  ///           uint16_t w = qs[2*ib32*... + l] (2 LE byte loads);   // DELTA(b): 9-bit
  ///           int idx   = w & 511;                                 // DELTA(a): 512
  ///           int signs = tcrv_iq2xs_ksigns[w >> 9];
  ///           vsetvl_e8m1(8); grid = vle8_i8m1(grid_i8 + idx*8); q8v = vle8_i8m1(q8);
  ///           m = vmsne(vand(vmv(signs), kmask), 0);          // sign-bit mask
  ///           g = vmerge(grid, vneg(grid), m);                // apply signs
  ///           p = vwmul_i16m2(g, q8v);  sumi = vwredsum(p, sumi);  q8 += 8;
  ///         }
  ///         bsum += sumi * (h ? ls2 : ls1);    // DELTA(c): explicit two-scale split
  ///       }
  ///     }
  ///     sumf = sumf + d * (float)bsum;                     // ONE emitc.expression
  ///   }
  ///   *s = 0.125f * sumf;                                  // SEPARATE statement
  /// q8 advances 8 each group CONTINUOUSLY across the half boundary (the {0,1} and
  /// {2,3} halves only split the sumi/bsum bookkeeping, not the activation cursor).
  /// The grid lookup is an indexed vle8(8) over a pointer (grid_i8 + idx*8); no
  /// vrgather table-index legality fact, no m1 table anchor. The integer chained
  /// vwredsum reduction is order-free; the per-super-block fp32 fold is invoked in
  /// STRICT ascending order so fp non-associativity is byte-exact across all
  /// -ffp-contract modes; the trailing 0.125f multiply stays OUT of the accumulate
  /// expression.
  mlir::LogicalResult emitIQ2XSQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq2_s_q8_K super-block dot-product for one
  /// tcrv_rvv.iq2_s_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no raw()).
  /// It is the THIRD member of the deep IQ tail and the SIBLING of
  /// emitIQ2XSQ8KBlockDot -- it REUSES the GRID + sign mechanism (indexed
  /// packed-uint64 GRID lookup, the kmask sign-bit plane + broadcast/vand/vmsne/vneg/
  /// vmerge per-lane sign apply, the signed widening product + chained vwredsum, the
  /// integer bsum fold, the explicit per-sub-block scales[] two-half split, the
  /// per-super-block fp32 fold, the 0.125f factor) with THREE structural deltas vs
  /// iq2_xs, mirroring _generic (quants.c:947-997):
  ///   static const int64_t tcrv_iq2s_grid[1024] = { ... };  // 1024-entry GRID
  ///   static const uint8_t tcrv_iq2s_kmask[8] = {1,2,4,8,16,32,64,128};
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_iq2s_kmask, 8);   // ONCE
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq2s_grid;
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*82;  const uint8_t *yb = vy + ibl*292;
  ///     float d = (float)*(const _Float16 *)(xb) * *(const float *)(yb);
  ///     const uint8_t *qs = xb + 2;   const uint8_t *sgn = xb + 34;   // signs=qs+32
  ///     const uint8_t *qh = xb + 66;  const uint8_t *sc = xb + 74;
  ///     const int8_t *q8 = yb + 4;    int32_t bsum = 0;
  ///     for (size_t ib32 = 0; ib32 < 8; ++ib32) {        // FLAT unrolled
  ///       int ls1 = 2*(sc[ib32] & 0xf) + 1;  int ls2 = 2*(sc[ib32] >> 4) + 1;
  ///       int qhb = qh[ib32];   // DELTA(b): the per-sub-block qh-bit plane byte
  ///       // half h=0: groups l=0,1 with ls1; half h=1: groups l=2,3 with ls2.
  ///       for (h = 0..1) {
  ///         int32_t sumi = 0;  (chained i32m1 reduction seed)
  ///         for (l = 2*h .. 2*h+1) {  // each group of 8 elements
  ///           // DELTA(a)+(b): idx = qs[ib32*4 + l] | ((qhb << (8-2*l)) & 0x300):
  ///           int idx = qs[ib32*4 + l] | ((qhb << (8 - 2*l)) & 0x300);  // 10-bit
  ///           // DELTA(c): signs read DIRECTLY from the explicit sign region:
  ///           int signs = sgn[ib32*4 + l];  // NO ksigns lookup
  ///           vsetvl_e8m1(8); grid = vle8_i8m1(grid_i8 + idx*8); q8v = vle8_i8m1(q8);
  ///           m = vmsne(vand(vmv(signs), kmask), 0);          // sign-bit mask
  ///           g = vmerge(grid, vneg(grid), m);                // apply signs
  ///           p = vwmul_i16m2(g, q8v);  sumi = vwredsum(p, sumi);  q8 += 8;
  ///         }
  ///         bsum += sumi * (h ? ls2 : ls1);    // explicit two-scale split
  ///       }
  ///     }
  ///     sumf = sumf + d * (float)bsum;                     // ONE emitc.expression
  ///   }
  ///   *s = 0.125f * sumf;                                  // SEPARATE statement
  /// q8 advances 8 each group CONTINUOUSLY across the half boundary. The grid lookup
  /// is an indexed vle8(8) over a pointer (grid_i8 + idx*8); no vrgather table-index
  /// legality fact, no m1 table anchor. The integer chained vwredsum reduction is
  /// order-free; the per-super-block fp32 fold is invoked in STRICT ascending order so
  /// fp non-associativity is byte-exact across all -ffp-contract modes; the trailing
  /// 0.125f multiply stays OUT of the accumulate expression. The qs index byte
  /// (`qs[ib32*4+l]`, a single byte) and the qh-bit injection (`(qhb<<(8-2*l))&0x300`)
  /// give the 10-bit grid index; the sign byte is read DIRECTLY from the sign region
  /// at qs+32 (iq2_s has NO ksigns plane).
  mlir::LogicalResult emitIQ2SQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq1_s_q8_K super-block dot-product for one
  /// tcrv_rvv.iq1_s_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no raw()).
  /// It is the TERNARY class -- the last common ggml dot kernel. It REUSES the
  /// grid-codebook indexed-lookup integer core from the iq2/iq3 siblings (packed
  /// uint64 GRID, vle8(8) over grid_i8 + idx*8, vwmul_i16m2 product + chained
  /// vwredsum reduction, the per-super-block fp32 fold) but introduces a GENUINELY NEW
  /// mechanism -- the per-block DELTA term -- mirroring _generic (quants.c:1099-1140):
  ///   static const int64_t tcrv_iq1s_grid[2048] = { ... };  // 2048 TERNARY entries
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq1s_grid;
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*50;  const uint8_t *yb = vy + ibl*292;
  ///     float d = (float)*(const _Float16 *)(xb) * *(const float *)(yb);
  ///     const uint8_t  *qs = xb + 2;       const uint16_t *qh = xb + 34;
  ///     const int8_t   *q8 = yb + 4;       const int16_t *bsums = yb + 260;
  ///     int32_t sumi = 0;  int32_t sumi1 = 0;
  ///     for (size_t ib = 0; ib < 8; ++ib) {           // FLAT unrolled
  ///       int qhw   = qh[ib];                          // a uint16 load
  ///       int ls    = 2*((qhw >> 12) & 7) + 1;         // qh bits 12..14
  ///       int delta = 1 - 2*((qhw >> 15) & 1);         // qh bit 15 -> +1/-1
  ///       int32_t lacc = 0;  (chained i32m1 reduction seed)
  ///       for (l = 0..3) {                             // each group of 8 elements
  ///         int idx = qs[ib*4 + l] | (((qhw >> (3*l)) & 7) << 8);   // 11-bit
  ///         vsetvl_e8m1(8);  grid = vle8_i8m1(grid_i8 + idx*8);     // ternary, signed
  ///                          q8v  = vle8_i8m1(q8);
  ///         p = vwmul_i16m2(grid, q8v);  lacc = vwredsum(p, lacc);  q8 += 8;  // NO signs
  ///       }
  ///       int lsum = __riscv_vmv_x_s_i32m1_i32(lacc);
  ///       sumi  += ls * lsum;                                         // grid dot
  ///       sumi1 += ls * delta * (bsums[2*ib+0] + bsums[2*ib+1]);      // DELTA term
  ///     }
  ///     sumf = sumf + d * ((float)sumi + 0.125f * (float)sumi1);      // ONE expr
  ///   }
  ///   *s = sumf;                                       // NO trailing factor
  /// The grid is byte-viewed as `const int8_t *` so the 0xff ternary bytes read as the
  /// signed value -1 (byte-identical to ggml's own grid read); there is NO sign plane,
  /// NO kmask, NO vmsne/vand/vneg/vmerge. The 0.125 (IQ1S_DELTA) is applied EXACTLY
  /// ONCE per super-block, inside the fold, to (float)sumi1 only; the inner add stays
  /// in the emitc.expression tree so the compiler contracts d*((float)sumi + 0.125f*
  /// (float)sumi1) identically to ggml's d*(sumi + IQ1S_DELTA*sumi1) under
  /// -ffp-contract=fast. The integer chained vwredsum reduction is order-free; the
  /// per-super-block fp32 fold is invoked in STRICT ascending order so fp
  /// non-associativity is byte-exact across all -ffp-contract modes. The store is the
  /// bare accumulator *s = sumf (NO 1/8 or 1/4 global factor).
  mlir::LogicalResult emitIQ1SQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_iq1_m_q8_K super-block dot-product for one
  /// tcrv_rvv.iq1_m_q8_k_block_dot op as fully STRUCTURED emitc nodes (I5; no raw()).
  /// It is the LAST common ggml dot kernel -- the iq1_s sibling completing 100% ggml
  /// dot coverage. It REUSES the SAME 2048-entry iq1s_grid ternary codebook lookup
  /// (vle8(8) over grid_i8 + idx*8, vwmul_i16m2 signed product, vwredsum reduce -- the
  /// grid bytes are themselves signed, NO sign plane) and the per-block DELTA
  /// mechanism (IQ1M_DELTA = 0.125), mirroring _generic (quants.c:1142-1201) with three
  /// NEW pieces:
  ///   static const uint64_t tcrv_iq1m_grid[2048] = { ... };  // SAME ternary grid
  ///   float sumf = 0.0f;  size_t nb = n / 256;
  ///   const int8_t *grid_i8 = (const int8_t *)tcrv_iq1m_grid;
  ///   for (size_t ibl = 0; ibl < nb; ibl += 1) {
  ///     const uint8_t *xb = vx + ibl*56;  const uint8_t *yb = vy + ibl*292;
  ///     const uint8_t  *qs = xb + 0;  const uint8_t *qh = xb + 32;
  ///     const uint16_t *sc = (const uint16_t *)(xb + 48);
  ///     const int8_t   *q8 = yb + 4;
  ///     // (a) RECONSTRUCT the packed iq1m_scale fp16 super-block scale (a bit
  ///     //     reinterpret, NOT a numeric conversion -- mirroring ggml's union):
  ///     uint16_t scbits = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00)
  ///                       | (sc[3]&0xf000);
  ///     float d = (float)*(const _Float16 *)(&scbits) * *(const float *)(yb);
  ///     int32_t sumi1 = 0;  int32_t sumi2 = 0;
  ///     for (size_t ib = 0; ib < 8; ++ib) {        // FLAT unrolled
  ///       int qh0 = qh[2*ib + 0];  int qh1 = qh[2*ib + 1];
  ///       int delta[4] = { qh0&8?-1:1, qh0&0x80?-1:1, qh1&8?-1:1, qh1&0x80?-1:1 };
  ///       int sum1_0=0, sum1_1=0, sum2_0=0, sum2_1=0;
  ///       for (l = 0..3) {                          // each group of 8 elements
  ///         int qhl = (l<2)?qh0:qh1;  int sh = (l%2)? 4 : 8;  // index-high shift
  ///         int idx = qs[ib*4 + l] | ((qhl << sh) & 0x700);   // 11-bit
  ///         vsetvl_e8m1(8);  grid=vle8(grid_i8+idx*8);  q8v=vle8(q8);
  ///         p = vwmul_i16m2(grid, q8v);  lsum1 = vwredsum(p, seed0);    // grid dot
  ///         s = vwredsum_i8m1_i16m1(q8v, seed0);  lsum2 = vmv_x_s(s);   // Σq8 (NEW)
  ///         sum1[l/2] += lsum1;  sum2[l/2] += lsum2 * delta[l];  q8 += 8;
  ///       }
  ///       int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1;  // groups 0..1
  ///       int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1;  // groups 2..3
  ///       sumi1 += sum1_0*ls1 + sum1_1*ls2;
  ///       sumi2 += sum2_0*ls1 + sum2_1*ls2;
  ///     }
  ///     sumf = sumf + d * ((float)sumi1 + 0.125f * (float)sumi2);  // ONE expr
  ///   }
  ///   *s = sumf;                                    // NO trailing factor
  /// The per-group delta (FOUR independent signs) means the q8 16-element bsums region
  /// is UNUSABLE; instead each group reduces a fresh Σq8 (i8m1 -> i16m1 vwredsum) of
  /// the SAME q8 vector loaded for the grid dot. The integer dot/accumulation is
  /// order-free; the per-super-block fp32 fold is invoked in STRICT ascending order so
  /// fp non-associativity is byte-exact across all -ffp-contract modes. The store is
  /// the bare accumulator *s = sumf.
  mlir::LogicalResult emitIQ1MQ8KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

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

  /// Emit the COMPLETE ggml ggml_vec_dot_q1_0_q8_0 block dot-product for one
  /// tcrv_rvv.q1_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-control-flow blob -- every value is a node in the IR graph). It
  /// is the BINARY ({-1,+1}) class: each q1_0 weight bit is a SIGN (set -> +q8,
  /// clear -> -q8) and the q8 value itself is the magnitude. It REUSES the
  /// iq2_xxs kmask/vmsne sign-plane primitive (the {1,2,4,...,128} bit-selector +
  /// vmsne != 0 -> a per-lane sign mask) over 8-lane groups, but with NO codebook,
  /// NO nibble unpack, NO offset-binary `-8` bias. The structured shape is:
  ///   static const uint8_t tcrv_q1_0_kmask[8] = {1,2,4,8,16,32,64,128};
  ///   float sumf = 0.0f;  size_t nb = n / 128;
  ///   vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_q1_0_kmask, 8);   // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;
  ///     float d0 = (float)*(const _Float16 *)(xb);
  ///     float sumi = 0.0f;
  ///     for (k = 0; k < 4; ++k) {                      // UNROLLED sub-blocks
  ///       const uint8_t *yb = vy + (ib*4 + k)*34;
  ///       float d1 = (float)*(const _Float16 *)(yb);
  ///       int32_t sumi_block = 0;  size_t vl = __riscv_vsetvl_e8m1(8);
  ///       for (b = 0; b < 4; ++b) {                    // UNROLLED 8-lane groups
  ///         uint8_t bits = *(const uint8_t *)(xb + 2 + k*4 + b);
  ///         vuint8m1_t m_bits = __riscv_vand_vv_u8m1(
  ///             __riscv_vmv_v_x_u8m1(bits, vl), kmask, vl);
  ///         vbool8_t   m = __riscv_vmsne_vx_u8m1_b8(m_bits, 0, vl);
  ///         vint8m1_t  q8 = __riscv_vle8_v_i8m1(yb + 2 + b*8, vl);
  ///         vint16m2_t q8w = __riscv_vwcvt_x_x_v_i16m2(q8, vl);  // widen FIRST
  ///         vint16m2_t w = __riscv_vmerge_vvm_i16m2(
  ///             __riscv_vneg_v_i16m2(q8w, vl), q8w, m, vl);  // +q8 if bit set
  ///         vint32m1_t seed = __riscv_vmv_v_x_i32m1(sumi_block, 1);
  ///         sumi_block = __riscv_vmv_x_s_i32m1_i32(
  ///             __riscv_vwredsum_vs_i16m2_i32m1(w, seed, vl));
  ///       }
  ///       sumi = sumi + d1 * (float)sumi_block;          // ggml exact order
  ///     }
  ///     sumf = sumf + d0 * sumi;                          // ggml exact order
  ///   }
  ///   *s = sumf;
  /// The q8 group is WIDENED to i16 before the negate/merge so the sign of the
  /// int8 boundary value -128 (+128) is representable -- an i8 vneg of -128
  /// overflows back to -128, but ggml's _generic negates in the int domain, so the
  /// i16-domain decode is byte-exact over the FULL int8 range (not only ggml's
  /// [-127,127] q8 domain). The 8-lane sign-plane groups keep the core VLEN-robust
  /// (8 <= e8m1 VLMAX at any VLEN >= 64; rv64gcv guarantees >= 128). The
  /// block-format facts are the op's typed attrs (I4 mirror); the emission is the
  /// op's fixed structure. The five fp16 reads (d0 + d1_{0..3}) are the only
  /// sanctioned opaque scalar pieces.
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

  /// Emit the COMPLETE ggml ggml_vec_dot_q4_K_q8_K block kernel (the q4_K K4b
  /// increment) for one tcrv_rvv.q4_k_q8_k_block_dot op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob). It reuses the SAME K4a
  /// super-block integer core (emitQ4_KSuperBlockAux32Core: the 4-bit unpack
  /// into aux8[256] + the 6-bit scale/min bit-dance + the nested sub-block
  /// uint6-scaled i32 accumulation, returning the per-super-block aux32[8] AND
  /// the decoded scales/mins pointer) and adds the DEFERRED two-level fp32 fold
  /// (the q6_K K2 mechanism), the q4_K MIN term, and the fp32 *s store, mirroring
  /// _generic (quants.c:711-717) so byte-exactness is by construction:
  ///   float sums[8]; vfloat32m2_t sumsv = vfmv_v_f_f32m2(0.0f, 8); // ONCE
  ///   float sumf = 0.0f;                                            // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*144;  const uint8_t *yb = vy + ib*292;
  ///     <K4a integer core -> aux32, scalesU8 (mins = scalesU8 + 8)>;
  ///     float dy = *(const float *)(yb + 0);          // fp32 activation d, ONCE
  ///     // ---- the MIN term (q4_K-specific) ----
  ///     int sumi = 0;
  ///     for (size_t j = 0; j < 16; ++j)
  ///       sumi += (int)bsums[j] * (int)mins[j/2];     // bsums int16 SIGN-EXT
  ///     // ---- the deferred fp32 positive fold (q6_K K2) ----
  ///     float d = (float)*(const _Float16 *)(xb+0) * dy;
  ///     vfloat32m2_t af = vfcvt_f_x_v_f32m2(aux32, 8);
  ///     vfloat32m2_t pr = vfmul_vf_f32m2(af, d, 8);    // SEPARATE mul (NOT fma)
  ///     sumsv = vfadd_vv_f32m2(sumsv, pr, 8);          // SEPARATE add (NOT fma)
  ///     // ---- the MIN subtraction (single emitc.expression) ----
  ///     float dmin = (float)*(const _Float16 *)(xb+2) * dy;
  ///     sumf = sumf - dmin * (float)sumi;              // ONE C statement
  ///   }
  ///   float sums8[8];  vse32_v_f32m2(sums8, sumsv, 8); // lane l -> sums8[l]
  ///   sumf += sums8[0]; ...; sumf += sums8[7];         // SEQUENTIAL l=0..7
  ///   *s = sumf;
  /// The fp32 byte-exactness pivots match q6_K K2 EXACTLY for the positive fold:
  /// the 8 lanes are independent, the d-multiply is a SEPARATE vfmul then a
  /// SEPARATE vfadd (NEVER fused), and the final horizontal sum is SEQUENTIAL
  /// ascending l=0..7. The NEW q4_K piece is the MIN term: the integer
  /// sumi = Σ_{j=0..15} bsums[j] * mins[j/2] (bsums int16 SIGN-extended, each
  /// decoded uint6 min spanning TWO consecutive bsums via j/2, integer/order-free)
  /// then `sumf -= dmin * sumi` carried in the SCALAR sumf accumulator IN-LOOP
  /// (in super-block order) -- distinct from q6_K's post-loop-only sumf -- and
  /// summed with the 8 lanes only AFTER the loop. The MIN subtraction is ONE
  /// emitc.expression so it renders as ggml's single C statement. The
  /// block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitQ4_KQ8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_q5_K_q8_K block kernel (the q5_K
  /// COVERAGE rung) for one tcrv_rvv.q5_k_q8_k_block_dot op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob, no raw()). q5_K is q4_K's
  /// K4b emission EXACTLY -- the SAME shared super-block integer core
  /// (emitQ4_KSuperBlockAux32Core: the 4-bit unpack into aux8[256] + the 6-bit
  /// scale/min bit-dance + the per-sub-block uint6-scaled i32 accumulation), the
  /// SAME deferred two-level fp32 fold, the SAME q4_K MIN term, the SAME
  /// sequential horizontal sum and *s store -- with the ONLY new piece being the
  /// qh 5th-bit injection plumbed into the shared core via cx.hasQh / cx.qhOffset
  /// (each unpacked nibble gets +16 added in the u8 domain when its qh plane bit
  /// is set, lifting q4 in [0,15] to q5 in [0,31], mirroring _generic
  /// quants.c:756-764's `a[l] += (hm[l] & m ? 16 : 0); m <<= 1` per 32-element
  /// half). The block-format facts (now including the qh @16 offset, the stride
  /// 176, and qs @48) are the op's typed attrs (I4 mirror). Because the fold/min/
  /// horizontal-sum are byte-identical to K4b and the core is shared, only the
  /// gated qh inject distinguishes the emitted nodes.
  mlir::LogicalResult emitQ5_KQ8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_q2_K_q8_K block kernel (the q2_K
  /// COVERAGE rung) for one tcrv_rvv.q2_k_q8_k_block_dot op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob, no raw()). q2_K reuses the
  /// super-block scaffolding + the scale+min structure of q4_K but is genuinely
  /// SIMPLER in three respects, mirroring _generic (quants.c:514-564)
  /// line-for-line so byte-exactness is by construction:
  ///   int8_t aux8[256];  float sumf = 0.0f;                       // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*84;  const uint8_t *yb = vy + ib*292;
  ///     // (A) 2-bit weight unpack: for each 32-byte qs chunk (k in 0..1) and
  ///     //     each shift in {0,2,4,6}, aux8[128*k + 32*(shift/2) + l] =
  ///     //     (qs[k*32+l] >> shift) & 3 (the 32 lanes l) -- u8m2 load + vsrl +
  ///     //     vand + u8->i8 reinterpret + vse8. Sub-block s -> aux8[16s:16s+16].
  ///     // (B) per-sub-block: scale = sc[s] & 0xF, min = sc[s] >> 4 (the SIMPLE
  ///     //     4-bit nibbles, NO utmp/kmask bit-dance).
  ///     int isum = 0; int summs = 0;
  ///     for (size_t s = 0; s < 16; ++s) {
  ///       int sc = (int)*(const uint8_t *)(xb + s);
  ///       // isuml = Σ_{l=0..15} q8[16s+l] * aux8[16s+l]  (vector widen-reduce)
  ///       int isuml = vmv_x_s(vwredsum(vwmul(q8v, av), seed0));
  ///       isum  += (sc & 0xF) * isuml;
  ///       summs += (int)bsums[s] * (sc >> 4);
  ///     }
  ///     // (C) the SCALAR fp32 fold, ONE C statement:
  ///     float dy = *(const float *)(yb + 0);
  ///     float dall = (float)*(const _Float16 *)(xb + 80) * dy;
  ///     float dmin = (float)*(const _Float16 *)(xb + 82) * dy;
  ///     sumf += dall * isum - dmin * summs;
  ///   }
  ///   *s = sumf;
  /// The integer side is order-free (associative int add) so the per-sub-block
  /// isuml is computed in the vector domain; the ONLY pinned order is the SCALAR
  /// fp32 fold `sumf += dall*isum - dmin*summs` carried in super-block order
  /// (distinct from q4_K's 8-lane deferred vector + post-loop horizontal sum --
  /// q2_K's positive term is a single scalar isum, so NO sums vector exists). The
  /// fold is ONE emitc.expression so the two products + the add + the subtract
  /// render as ggml's single C statement (quants.c:561) and track the
  /// contraction. The block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitQ2_KQ8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

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

  /// Emit the COMPLETE ggml ggml_vec_dot_q6_K_q8_K block kernel (the K-quant K2
  /// increment) for one tcrv_rvv.q6_k_q8_k_block_dot op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob). It reuses the SAME K1
  /// super-block integer core (emitQ6_KSuperBlockAux32Core: the 6-bit unpack into
  /// aux8[256] + the nested sub-block int8-scaled i32 accumulation, returning the
  /// per-super-block aux32[8]) and adds the DEFERRED two-level fp32 fold + the
  /// fp32 *s store, mirroring _generic (quants.c:813-852) so byte-exactness is by
  /// construction:
  ///   float sums[8];  vfloat32m2_t sumsv = vfmv_v_f_f32m2(0.0f, 8);  // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*210;  const uint8_t *yb = vy + ib*292;
  ///     vint32m2_t aux32 = <K1 integer core>;          // RESET per super-block
  ///     float d = (float)*(const _Float16 *)(xb+208) * *(const float *)(yb+0);
  ///     vfloat32m2_t af = vfcvt_f_x_v_f32m2(aux32, 8); // (float)aux32[l], RNE
  ///     vfloat32m2_t pr = vfmul_vf_f32m2(af, d, 8);    // SEPARATE mul (NOT fma)
  ///     sumsv = vfadd_vv_f32m2(sumsv, pr, 8);          // SEPARATE add (NOT fma)
  ///   }
  ///   float sums8[8];  vse32_v_f32m2(sums8, sumsv, 8); // lane l -> sums8[l]
  ///   float sumf = sums8[0]; sumf += sums8[1]; ...; sumf += sums8[7]; // l=0..7
  ///   *s = sumf;
  /// The fp32 byte-exactness pivots: the 8 lanes are INDEPENDENT (so vectorizing
  /// the fold body is byte-safe), the d-multiply is a SEPARATE vfmul then a
  /// SEPARATE vfadd (NEVER a fused vfmacc/vfmadd -- that diverges from _generic's
  /// scalar mul/add under -ffp-contract=off, the INC-2a FMA class), and the final
  /// cross-lane horizontal sum is SEQUENTIAL ascending l=0..7 via 8 scalar
  /// emitc.add (NEVER a vector vfredusum -- fp add is non-associative). Lane l
  /// <-> index l is preserved by the vse32 store and pinned by the K1 aux32[l]
  /// <-> l ordering. The block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitQ6_KQ8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const;

  /// Emit the COMPLETE ggml ggml_vec_dot_q3_K_q8_K block kernel (the q3_K
  /// COVERAGE rung, the LAST common K-quant) for one tcrv_rvv.q3_k_q8_k_block_dot
  /// op as fully STRUCTURED emitc nodes (I5; no verbatim C-control-flow blob, no
  /// raw()). q3_K COMPOSES mechanisms already in hand, mirroring _generic
  /// (quants.c:566-643) line-for-line so byte-exactness is by construction:
  ///   int8_t aux8[256];  int8_t sc16[16];                          // scratch
  ///   float sums8[8];  vfloat32m2_t sums = vfmv_v_f_f32m2(0.0f, 8); // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*110;  const uint8_t *yb = vy + ib*292;
  ///     // (A) 2-bit + SUBTRACTIVE-hmask unpack into aux8[256] (signed [-4,3]):
  ///     //     for each 32-byte qs chunk (chunk 0..1) and each shift in {0,2,4,6}
  ///     //     (shiftIdx 0..3), with bit position p = 4*chunk + shiftIdx:
  ///     //       aux8[128*chunk + 32*shiftIdx + l] =
  ///     //         (((qs>>shift)&3) | (((hm[l]>>p)&1)<<2)) - 4
  ///     //     -- u8m2 load qs + vsrl/vand for low 2 bits; vsrl/vand/vsll of the
  ///     //     SAME hmask plane (loaded once per chunk) for the high bit, OR-ed
  ///     //     in (vor); u8->i8 reinterpret; vsub 4; vse8. hm is NEVER advanced.
  ///     // (B) the SIGNED 6-bit scale dance: read scales[12] as 3 u32 words,
  ///     //     q3_K shuffle (kmask1=0x03030303, kmask2=0x0f0f0f0f) into 16 bytes
  ///     //     read as SIGNED int8; scale_j = (int)sc16[j] - 32.
  ///     vint32m2_t aux32 = 0;                          // RESET per super-block
  ///     for (size_t js = 0; js < 16; ++js) {           // q6_K-style sub-loop
  ///       int scale = (int)sc16[js] - 32;
  ///       // two 8-halves: vwmul i8xi8->i16, vwmacc.vx aux32 += scale*i16
  ///     }
  ///     // (C) the NO-min deferred fp32 fold (q6_K's, identical):
  ///     float d = (float)*(const _Float16 *)(xb+108) * *(const float *)(yb+0);
  ///     vfloat32m2_t af = vfcvt_f_x_v_f32m2(aux32, 8); // (float)aux32[l], RNE
  ///     vfloat32m2_t pr = vfmul_vf_f32m2(af, d, 8);    // SEPARATE mul (NOT fma)
  ///     sums = vfadd_vv_f32m2(sums, pr, 8);            // SEPARATE add (NOT fma)
  ///   }
  ///   float sums8[8];  vse32_v_f32m2(sums8, sums, 8);  // lane l -> sums8[l]
  ///   float sumf = 0.0f; sumf += sums8[0]; ...; sumf += sums8[7]; // l=0..7
  ///   *s = sumf;
  /// The integer side is order-free; the pinned order is the q6_K NO-min fold
  /// (SEPARATE vfmul/vfadd, NEVER fma; SEQUENTIAL ascending horizontal sum, NEVER
  /// a vfredusum). There is NO min term and NO dmin (q3_K is symmetric, like
  /// q6_K). The block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitQ3_KQ8_KBlockDot(
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
