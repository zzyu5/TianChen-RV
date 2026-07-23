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

// Typed q3/high-mask and ternary repack artifact consumers.

// q3_K x q8_K 16x1-REPACKED GEVM (decode) body leaf, from the FRONT DOOR. q3_K is the
// LAST (and most intricate) K-quant repack sibling: a FAITHFUL reuse of the
// oracle-verified q6_K SINGLE-accumulator no-min scaffold (emitRepackKQuantGemvBodyQ6K
// -- LANE-WISE vwmacc block-as-lane strips, 16 SIGNED int8 scales vle8_v_i8 +
// vsext_vf2_i16, a SINGLE super-block d fold with NO dmin / NO bsums / NO min term) with
// ONE q3_K-specific delta: the 3-BIT SUBTRACTIVE-HMASK weight assembly. Each weight is
// `((qs >> shift) & 3) - ((hmask & (1<<p)) ? 0 : 4)` -- a 2-bit `qs` low plane (4 weights
// per byte at shift {0,2,4,6}) | a SINGLE `hmask` high bit lifted to bit-2, then the -4
// SUBTRACTIVE bias (bit CLEAR -> subtract 4, folded LANE-WISE via vsub_vx_i8 by 4). The
// hmask bit position `p = 4*(n/128) + shift_group` reuses the SAME 32-byte hmask plane
// across BOTH super-halves (the hmask byte index is NEVER advanced by super-half; only
// qs is). The 16 SIGNED 6-bit scales are PRE-UNPACKED + -32-biased at repack time, so the
// scale side is byte-identical to q6_K. Because |w| <= 4 (vs q6_K's [-32,31]), a full
// 16-position sub-block partial stays within int16 (16*4*127 = 8128 < 32767), so q3_K
// runs ONE i16 partial per sub-block with NO 2x8 k-chunk split. The byte-exact body of
// the RETIRED monolithic direct emitter emitRepackGemvQ3KQ8K, refactored to take the
// mapped ABI values + block-format facts as PARAMETERS (no monolith-op lookup, no
// trailing unused-result token); called ONLY from emitTypedRepackGemvLoopBody's K-quant
// no-min branch, gated on the in-region weft_rvv.repack_gemv_kquant_core anti-bypass
// brick (decode_model "q3_K"). Oracle-verified vs an INDEPENDENT scalar q3_K
// dequant-matmul reference (controls HMASK-OFF / HMASK-INV / SCALE-ROT / BIAS-OFF).
// RESULT-LESS (no monolith token). The hmask SECOND weight plane rides the SHARED qh
// slot (weightHmaskOffset == the loop op's weight_qh_byte_offset).

// [QH-MASK] SHARED native-interleaved-static mask-source decode helper. The ONE
// implementation of the native-mask single-bit-plane bias fuse, called by the q3_K
// GEVM leaf, the q3_K GEMM leaf, and the q5_K super-block nibble unpack.
//
// [档 C#6 归因订正] The single-bit-plane restriction is enforced BY CONSTRUCTION, NOT
// by a runtime `bit_plane_width==1` predicate. The helper isolates EXACTLY one bit
// (`1 << bitPos`, a single `bitPos` param), so a 2-bit high plane (q6_K's 0x03 qh)
// CANNOT be expressed through it. q6_K is therefore excluded by EMITTER-IDENTITY
// dispatch: the K-quant no-min GEVM/GEMM arms route decode_model "q6_K" to its own
// 2-bit two-plane leaf (emitRepackKQuantGemvBodyQ6K / ...Gemm...) and decode_model
// "q3_K" (single-bit hmask) to the leaf that calls this helper -- the discriminant is
// the decode_model WHAT, NEVER a codified/evaluated bit-plane-width attribute. The
// earlier "bit_plane_width==1 predicate term" wording overstated this as a codified
// predicate; it is documentary rationale for WHY the single-bit formats (q3_K hmask,
// q5_K qh) share this helper while q6_K's 2-bit plane does not.
mlir::Value VariantToEmitCFunc::emitNativeMaskStaticBitBias(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value base, mlir::Value maskSrcU8, int bitPos, bool biasWhenBitSet,
    int biasImm, llvm::StringRef lmul, bool elemSigned, mlir::Type u8VecType,
    mlir::Type resultVecType, mlir::Value vl, llvm::StringRef opName,
    llvm::StringRef role) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
  // The vbool predicate width for an i8/u8<lmul> vector is N = 8/LMUL, the SAME
  // mapping the q5_0/q5_1 REDESIGN-B qh decode + the q3_K GEVM sibling use.
  unsigned maskBits = 16;
  if (lmul == "mf8") maskBits = 64;
  else if (lmul == "mf4") maskBits = 32;
  else if (lmul == "mf2") maskBits = 16;
  else if (lmul == "m1") maskBits = 8;
  else if (lmul == "m2") maskBits = 4;
  else if (lmul == "m4") maskBits = 2;
  else if (lmul == "m8") maskBits = 1;
  std::string bitsStr = std::to_string(maskBits);
  mlir::Type boolType =
      emitc::OpaqueType::get(ctx, ("vbool" + bitsStr + "_t"));
  std::string vandCallee = ("__riscv_vand_vx_u8" + lmul).str();
  // vmsne==0 -> TRUE where the bit is SET (bias rides SET lanes: q5_K +16);
  // vmseq==0 -> TRUE where the bit is CLEAR (bias rides CLEAR lanes: q3_K -4).
  std::string cmpCallee =
      (llvm::Twine("__riscv_") + (biasWhenBitSet ? "vmsne" : "vmseq") +
       "_vx_u8" + lmul + "_b" + bitsStr)
          .str();
  std::string signStr = elemSigned ? "i8" : "u8";
  std::string addMuCallee = ("__riscv_vadd_vx_" + signStr + lmul + "_mu").str();
  // isolate the STATIC high bit: hbit = maskSrcU8 & (1 << bitPos).
  mlir::Value isolated = emitOpaqueCallBuilt(
      rewriter, loc, u8VecType, vandCallee, opName, role,
      [&](mlir::OpBuilder &b,
          mlir::Location l) -> llvm::SmallVector<mlir::Value> {
        mlir::Value immV =
            rewriter
                .create<emitc::LiteralOp>(loc, immI32Type,
                                          std::to_string(1 << bitPos))
                .getResult();
        return {maskSrcU8, immV, vl};
      });
  // lift the isolated bit to a per-lane bool plane.
  mlir::Value mask = emitOpaqueCallBuilt(
      rewriter, loc, boolType, cmpCallee, opName, role,
      [&](mlir::OpBuilder &b,
          mlir::Location l) -> llvm::SmallVector<mlir::Value> {
        mlir::Value zero =
            rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0").getResult();
        return {isolated, zero, vl};
      });
  // FUSE the +/- bias into ONE masked add on exactly the selected lanes.
  return emitOpaqueCallBuilt(
      rewriter, loc, resultVecType, addMuCallee, opName, role,
      [&](mlir::OpBuilder &b,
          mlir::Location l) -> llvm::SmallVector<mlir::Value> {
        mlir::Value immV =
            rewriter
                .create<emitc::LiteralOp>(loc, immI32Type,
                                          std::to_string(biasImm))
                .getResult();
        return {mask, base, base, immV, vl};
      });
}

mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemvBodyQ3K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQsOffset, int64_t activationQuantOffset,
    int64_t weightScalesOffset, int64_t weightHmaskOffset, int64_t nSubblocks,
    int64_t weightInterleave, int64_t half, bool rolledMainTerm) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the 16-way
    // interleaved repack reads the SAME bytes either way). "mf2" (default, absent) is
    // the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2 (f16 scale m1),
    // running at half_lanes e16m1 lanes per strip. "m1" is the WHOLE-LMUL chain
    // RVV0.7.1 requires: the chain shifts up one notch, ONE 16-lane strip.
    llvm::StringRef l8 = coreLmul;                         // mf2 -> mf2; m1 -> m1
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";  // mf2 -> m1;  m1 -> m2
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";  // mf2 -> m2;  m1 -> m4
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    // The 3-bit weight lane is a SIGNED i8 in [-4,3] (-4 subtractive bias baked in);
    // the qs|hmask assembly runs on the UNSIGNED u8 raw value first, then vsub 4.
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked q3_K (K-quant super-block) GEVM block-format structural facts
    // (I4 mirror, PARAMETERS now -- the loop body op's pinned attrs, read by the K-quant
    // no-min branch of emitTypedRepackGemvLoopBody and passed in): QK_K=256,
    // block_q3_Kx16 weight stride 1824 (16 fp16 d + 256 signed int8 scales + 512 hmask
    // high-bit + 1024 qs low-2-bit), block_q8_K activation stride 292 (fp32 d + 256 int8
    // quants + 16 int16 bsums, bsums UNUSED), qs at +800, hmask at +288 (weightHmaskOffset,
    // the SHARED qh slot), signed scales at +32, activation int8 quants at +4, 16 weight
    // columns per group, 16 sub-blocks of 16, and the VLEN-derived e8 half width.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    (void)nSubblocks;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-q3_K output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width.
    mlir::Value vl8 = sizeLit(half);
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");

    // size_t nb = n / QK_K;   size_t nc_groups = nc / 16;
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load helpers ----
    // UNSIGNED u8 contiguous strip load (the qs/hmask planes).
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // SIGNED i8 contiguous strip load (the per-sub-block signed int8 scales).
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
    auto loadI8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // vand_vx_u8 / vsrl_vx_u8 -- the 3-bit qs base extract + hmask bit isolate.
    // (The OLD vsll_vx_u8 / vor_vv_u8 per-lane expand pair is RETIRED by the
    // native-mask KNEST recon below.)
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    auto u8Imm = [&](llvm::StringRef callee, mlir::Value v,
                     llvm::StringRef imm) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, imm)
                    .getResult();
            return {v, immV, vl8};
          });
    };
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    // ===== REDESIGN-B (native-mask KNEST) single-bit-plane hmask recon ==========
    // q3_K's hmask is a SINGLE-bit plane (the 3rd bit) -- the STRUCTURAL ANALOG of
    // q5_0/q5_1's qh 5th bit -- so the SAME stock native-mask trick applies: test the
    // high bit IN PLACE with vand(1<<p) + vmseq==0 (a per-lane bool plane), then FUSE
    // the high-bit select and the -4 SUBTRACTIVE bias into ONE masked op
    // vadd_vx_i8_mu(mask0, base, base, -4). This RETIRES the OLD per-lane expand chain
    // (vsrl(hm,p) | vand 0x01 | vsll 2 | vor | vsub 4). Byte-exact BY CONSTRUCTION:
    // mask0 TRUE <=> hmask bit p CLEAR <=> stock's `if (hbit==0) q -= 4` (be_q3k 0/8:
    // native === OLD === stock scalar). The native-mask + vbool-width work now rides
    // the SHARED [QH-MASK] emitNativeMaskStaticBitBias helper (biasWhenBitSet=false,
    // biasImm=-4), the SAME implementation the q3_K GEMM leaf + q5_K super-block call.
    // Assemble ONE 3-bit SUBTRACTIVE signed weight strip from a qs strip + the hmask
    // strip. `shift` = 2*shift_group (the 2-bit qs lane), `p` = the hmask bit
    // position 4*sh + shift_group. The 2-bit low plane (vand 0x03) reinterpreted to
    // signed i8 is the base; the SHARED helper isolates the SINGLE hmask bit p
    // (vand(1<<p) + vmseq==0, ONE bit -- NOT q6_K's 0x03 two-bit qh) + fuses the -4.
    auto assembleWeight = [&](mlir::Value qs, mlir::Value hm, int shift,
                              int p) -> mlir::Value {
      mlir::Value shifted =
          (shift == 0) ? qs : u8Imm(vsrlCallee, qs, std::to_string(shift));
      mlir::Value low2 = u8Imm(vandCallee, shifted, "0x03");
      mlir::Value baseI8 = reinterpretToI8(low2); // i8 in [0,3]
      return emitNativeMaskStaticBitBias(
          rewriter, loc, baseI8, hm, /*bitPos=*/p, /*biasWhenBitSet=*/false,
          /*biasImm=*/-4, l8, /*elemSigned=*/true, u8mf2Type, i8mf2Type, vl8,
          opName, role); // i8 in [-4,3]
    };
    // A scalar i8 read of the activation quant byte a.qs[k] (int8).
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    // vwmacc_vx i8->i16: acc += scalar_activation * weight_vec (the sub-block dot).
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    // vwmacc_vv i16->i32: sumi += signed_scale_strip(i16) * sub_block_partial(i16).
    std::string vwmaccVV32Callee = ("__riscv_vwmacc_vv_i32" + l32).str();
    auto vwmaccVV32 = [&](mlir::Value acc, mlir::Value scaleStrip,
                          mlir::Value sumiS) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVV32Callee,
                            mlir::ValueRange{acc, scaleStrip, sumiS, vl8}, opName,
                            role);
    };
    // vsext_vf2 i8 strip -> i16 strip: the SIGNED per-sub-block scale (NOT a
    // zero-extend) lifted to the i16 lane the vwmacc_vv consumes.
    std::string vsextCallee = ("__riscv_vsext_vf2_i16" + l16).str();
    auto liftScaleToI16 = [&](mlir::Value i8scale) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vsextCallee,
                            mlir::ValueRange{i8scale, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t baseByteOff,
                            int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      int64_t totalOff = baseByteOff + laneOff * 2;
      if (totalOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    auto widenF16 = [&](mlir::Value f16strip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                            mlir::ValueRange{f16strip, vl8}, opName, role);
    };
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    auto fmulScalar = [&](mlir::Value vec, mlir::Value scalar) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    auto seedI32 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvI32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
    auto seedI16 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i16m1Type, mvI16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    mlir::Value aBase = activationBase;

    // ===== Outer weight-COLUMN-GROUP loop: for (x = 0; x < nc/16; ++x) =====
    auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    mlir::LogicalResult status = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();

      // const uint8_t *b = vx + x*nb*1824;  (the q3_Kx16 column group base).
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      mlir::Value bGroup = rewriter.create<emitc::AddOp>(
          loc, weightPtrType, weightBase, bGroupOff);

      // vfloat32m2_t sumf_h = vfmv_v_f(0, half) per strip (carried across blocks).
      std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
      auto seedF32 = [&]() -> mlir::Value {
        return emitOpaqueCallBuilt(
            rewriter, loc, f32m2Type, fmvCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                      .getResult();
              return {zero, vl8};
            });
      };
      llvm::SmallVector<mlir::Value> sumfVar;
      for (int64_t h = 0; h < numHalves; ++h) {
        auto v = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(f32m2Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(loc, v, seedF32());
        sumfVar.push_back(v);
      }

      // ===== Inner contraction-BLOCK loop: for (l = 0; l < nb; ++l) =====
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard bg(rewriter);
        rewriter.setInsertionPointToStart(blockLoop.getBody());
        mlir::Value lvar = blockLoop.getInductionVar();

        // const uint8_t *bl = b + l*1824;   const uint8_t *al = a + l*292;
        step("weight_block_base");
        mlir::Value blOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, lvar, sizeLit(weightStride));
        mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                       bGroup, blOff);
        step("act_block_base");
        mlir::Value alOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, lvar, sizeLit(activationStride));
        mlir::Value al = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, aBase, alOff);

        // -- the activation super-block delta d_y = *(const float *)&al.d. q8_K
        // stores a FLOAT (4 bytes) at offset 0. q3_K has a SINGLE d scale (no dmin),
        // so this feeds the ONE fold term.
        mlir::Value aD = emitOpaqueCallBuilt(
            rewriter, loc, floatType, floatReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, floatPtrConstType, al)
                      .getResult();
              return {aDCast};
            },
            llvm::StringRef("act_scale_scalar"));

        // ===== Per-block i32 accumulator per strip (the SINGLE q3_K accumulator;
        // NO bsums/min accumulator). =====
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Super-half loop: for (sh = 0; sh < QK_K/128; ++sh) =====
        // Each super-half is 2 sub-halves of 16 (grp 0/1); for each grp the 4 shift-
        // quadrants (2-bit qs shift 0/2/4/6) are 4 DISTINCT sub-blocks. Load the 4
        // SIGNED scale strips, run one 16-position i16 partial per quadrant, and fold
        // scale-weighted to i32. qs advances 32 bytes per super-half; the hmask byte
        // index is the SAME 32-byte plane for BOTH super-halves (only bit p shifts).
        for (int64_t sh = 0; sh < nSuperHalves; ++sh) {
          for (int64_t grp = 0; grp < 2; ++grp) {
            // Load the 4 shift-quadrant SIGNED scale strips. The sub-block index for
            // (super-half sh, shift-quadrant q, sub-half grp) is sh*8 + q*2 + grp.
            step("signed_scale_unpack");
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> scaleVal(numHalves);
            for (int64_t h = 0; h < numHalves; ++h) {
              for (int q = 0; q < 4; ++q) {
                int64_t sIdx = sh * 8 + q * 2 + grp;
                mlir::Value s8 = loadI8Strip(
                    bl, weightScalesOffset + sIdx * 16 + h * half);
                scaleVal[h].push_back(liftScaleToI16(s8));
              }
            }

            // i16 partials: acc[h][q] over the 16 positions in this sub-block (NO 2x8
            // k-chunk split -- |w| <= 4 keeps 16*4*127 = 8128 < 32767).
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> acc(numHalves);
            if (rolledMainTerm) {
              // ---- ROLLED whole-K-nest main term ([GAP-EMIT-KQUANT-GEVM-TILE-
              // ROUNDTRIP] maturity lever): the dominant per-16-position inner l-loop is
              // emitted as ONE runtime emitc.for; the per-strip x per-shift-quadrant i16
              // partials acc[h][q] are carried as RESIDENT SSA-register VariableOps
              // (seeded ABOVE the loop, load-accumulate-store INSIDE it). The runtime
              // position l only shifts the weight/activation base pointers (bl+l*16,
              // al+l); the compile-time remainder rides the strip-load helper. BYTE-EXACT
              // to the unrolled emit by construction: the vwmacc16 integer accumulation
              // order (l ascending, then h, then shift-quadrant q) is IDENTICAL -- only the
              // loop is materialized instead of unrolled.
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> accVar(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                for (int q = 0; q < 4; ++q) {
                  auto v = rewriter.create<emitc::VariableOp>(
                      loc, emitc::LValueType::get(i16m1Type),
                      emitc::OpaqueAttr::get(ctx, ""));
                  rewriter.create<emitc::AssignOp>(loc, v, seedI16());
                  accVar[h].push_back(v);
                }
              auto lLoop = rewriter.create<emitc::ForOp>(
                  loc, sizeLit(0), sizeLit(16), sizeLit(1), /*bodyBuilder=*/nullptr);
              {
                mlir::OpBuilder::InsertionGuard ig(rewriter);
                rewriter.setInsertionPointToStart(lLoop.getBody());
                mlir::Value lv = lLoop.getInductionVar();
                // Weight strips advance 16 bytes / position (16-way interleave), the
                // scalar activation 1 byte / position: shift the base pointers by l.
                mlir::Value lv16 = rewriter.create<emitc::MulOp>(
                    loc, sizeType, lv, sizeLit(16));
                mlir::Value blP = rewriter.create<emitc::AddOp>(
                    loc, weightPtrType, bl, lv16);
                mlir::Value alP = rewriter.create<emitc::AddOp>(
                    loc, activationPtrType, al, lv);
                step("act_quant_addr");
                mlir::Value aQ[4];
                for (int q = 0; q < 4; ++q)
                  aQ[q] = i8Read(alP, activationQuantOffset + sh * 128 + q * 32 +
                                          grp * 16);
                for (int64_t h = 0; h < numHalves; ++h) {
                  step("weight_qs_hmask_addr");
                  mlir::Value qs = loadU8Strip(
                      blP, weightQsOffset + (sh * 32 + grp * 16) * 16 + h * half);
                  mlir::Value hm = loadU8Strip(
                      blP, weightHmaskOffset + (grp * 16) * 16 + h * half);
                  for (int q = 0; q < 4; ++q) {
                    int shift = 2 * q;
                    int p = 4 * static_cast<int>(sh) + q;
                    mlir::Value w = assembleWeight(qs, hm, shift, p);
                    mlir::Value cur =
                        rewriter.create<emitc::LoadOp>(loc, i16m1Type, accVar[h][q])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(loc, accVar[h][q],
                                                     vwmacc16(cur, aQ[q], w));
                  }
                }
              }
              for (int64_t h = 0; h < numHalves; ++h)
                for (int q = 0; q < 4; ++q)
                  acc[h].push_back(
                      rewriter.create<emitc::LoadOp>(loc, i16m1Type, accVar[h][q])
                          .getResult());
            } else {
              // ---- UNROLLED main term (default, register-resident full unroll) ----
              for (int64_t h = 0; h < numHalves; ++h)
                for (int q = 0; q < 4; ++q)
                  acc[h].push_back(seedI16());

              for (int64_t l = 0; l < 16; ++l) {
                // 4 shift-quadrant activations (scalar, shared across strips): global
                // element position sh*128 + q*32 + grp*16 + l.
                step("act_quant_addr");
                mlir::Value aQ[4];
                for (int q = 0; q < 4; ++q)
                  aQ[q] = i8Read(al, activationQuantOffset + sh * 128 + q * 32 +
                                         grp * 16 + l);
                for (int64_t h = 0; h < numHalves; ++h) {
                  step("weight_qs_hmask_addr");
                  // qs byte index sh*32 + grp*16 + l; hmask byte index grp*16 + l (the
                  // SAME 32-byte plane for both super-halves).
                  mlir::Value qs = loadU8Strip(
                      bl, weightQsOffset + (sh * 32 + grp * 16 + l) * 16 + h * half);
                  mlir::Value hm = loadU8Strip(
                      bl, weightHmaskOffset + (grp * 16 + l) * 16 + h * half);
                  for (int q = 0; q < 4; ++q) {
                    int shift = 2 * q;
                    int p = 4 * static_cast<int>(sh) + q;
                    mlir::Value w = assembleWeight(qs, hm, shift, p);
                    acc[h][q] = vwmacc16(acc[h][q], aQ[q], w);
                  }
                }
              }
            }
            // sumi += sum over 4 shift-quadrants of signed_scale * acc (i16->i32).
            step("scale_subblock_fold");
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value cur =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                      .getResult();
              for (int q = 0; q < 4; ++q)
                cur = vwmaccVV32(cur, scaleVal[h][q], acc[h][q]);
              rewriter.create<emitc::AssignOp>(loc, sumiVar[h], cur);
            }
          }
        }

        // ===== End-of-block SINGLE fold per strip: sumf += cvt(sumi) * (d_x*d_y).
        // NO min term (no vfnmsac): q3_K's -4 subtractive bias is inside each weight. =
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dStrip = loadF16Strip(bl, 0, h * half);
          mlir::Value d0 = fmulScalar(widenF16(dStrip), aD);
          mlir::Value sumiV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult();
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value afterMain = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {curF, cvtI32F32(sumiV), d0, vl8};
              });
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], afterMain);
        }
      }

      // Per-strip store vse32(s + x*16 + h*half, sumf_h, half).
      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      auto storeHalf = [&](mlir::Value sumfStrip, int64_t laneOff) {
        step("output_addr");
        mlir::Value x16 = rewriter.create<emitc::MulOp>(
            loc, sizeType, x, sizeLit(weightInterleave));
        mlir::Value totalOff = x16;
        if (laneOff != 0)
          totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, x16,
                                                   sizeLit(laneOff));
        mlir::Value dst = rewriter.create<emitc::AddOp>(loc, floatPtrType,
                                                        output, totalOff);
        mlir::Value sumfVal =
            rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfStrip).getResult();
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
      };
      for (int64_t h = 0; h < numHalves; ++h)
        storeHalf(sumfVar[h], h * half);
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    (void)i32m1Type;
    // RESULT-LESS (no monolith token): the typed_repack_gemv_loop_body region is
    // result-less (the repacked lane-wise q3_K GEVM sinks through the output pointer,
    // not an SSA vector), so unlike the retired direct emitter this body leaf seeds NO
    // dead i32m1 result token.
    return mlir::success();
  }

// q3_K x q8_K 16x1-REPACKED PREFILL GEMM (prefill) body leaf, from the FRONT DOOR
// (S6 STRIP-OUTER TILED). [G8 stage-3, board-remeasured 2026-07-14] The weight-STRIP
// loop is hoisted OUTSIDE the block loop (same lever as q6_K); board objdump total
// 20022->9237, spill vs1r 1065->93 / vl1r 1179->221, MACs identical (2176), byte-exact
// 27/27. Because q3_K's sub-block is 16 positions (2x q6_K's 8), the fully-UNROLLED
// main term is a 20933-line body that thrashes I$ + leaves residual spill => unrolled+S6
// cold ~0.79 (near-miss); the [ROLL] schedule (runtime l-loop, measured-beneficial for
// q3_K) collapses it to a 3021-line body => rolled+S6 cold ~1.40 WIN. [ROLL] is the
// orthogonal schedule axis (RVVRepackScheduleFormula), per-format measured (q6_K
// prefers unrolled). The q3_K prefill sibling of emitRepackKQuantGemvBodyQ3K: the SAME
// 3-bit SUBTRACTIVE-hmask signed-weight assembly (-4
// bias), the SAME 16 SIGNED 6-bit scales (pre-unpacked + -32-biased at repack), and
// the SAME SINGLE-accumulator no-min fold, with the weight decode AMORTIZED once per
// 16-weight group across the 4 (and, across the row loop, M) interleaved block_q8_Kx4
// activation columns. Oracle-verified vs an INDEPENDENT scalar q3_K dequant-matmul
// reference (controls HMASK-OFF / HMASK-INV / SCALE-ROT / BIAS-OFF, plus the x4
// interleave).
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemmBodyQ3K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQsOffset,
    int64_t activationQuantOffset, int64_t weightScalesOffset,
    int64_t weightHmaskOffset, int64_t nSubblocks, int64_t weightInterleave,
    int64_t activationInterleave, int64_t half, bool rolledMainTerm, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::StringRef l8 = coreLmul;
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked q3_K PREFILL GEMM block-format facts (I4 mirror, PARAMETERS now
    // -- the loop body op's pinned attrs, read by the K-quant no-min branch of
    // emitTypedRepackGemmLoopBody and passed in): QK_K=256, block_q3_Kx16 weight stride
    // 1824 (SAME weight ABI as the GEVM), block_q8_Kx4 activation stride 1168 (4 fp32 d +
    // 1024 int8 + 64 int16 bsums, bsums UNUSED), qs at +800, hmask at +288
    // (weightHmaskOffset, the SHARED qh slot), signed scales at +32, interleaved
    // activation quants at +16, 16 weight columns / 4 activation columns, 16 sub-blocks
    // of 16, VLEN-derived e8 half width.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2
    (void)nSubblocks;
    // RVV1.0 (fractional chain) folds all 4 activation columns per pass; RVV0.7.1
    // (whole-LMUL) folds ONE column per pass to keep the per-pass live set bounded.
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave; // 1 @rvv07; 4 @rvv1.0

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-q3_K output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    mlir::Value vl8 = sizeLit(half);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("row_group_count");
    mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, rowCount, sizeLit(activationInterleave));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
    auto loadI8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    auto u8Imm = [&](llvm::StringRef callee, mlir::Value v,
                     llvm::StringRef imm) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, imm)
                    .getResult();
            return {v, immV, vl8};
          });
    };
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    // [QH-MASK] q3_K GEMM leaf: the SAME 3-bit SUBTRACTIVE-hmask assembly as the
    // q3_K GEVM sibling, now routed through the SHARED native-mask helper. The
    // 2-bit low plane (vand 0x03) reinterpreted to signed i8 is the base; the
    // helper isolates the SINGLE hmask bit p (vand(1<<p) + vmseq==0) and fuses the
    // -4 SUBTRACTIVE bias into ONE vadd_vx_i8_mu -- byte-exact to the RETIRED OLD
    // per-lane expand chain (vsrl(hm,p) | vand 0x01 | vsll 2 | vor | vsub 4).
    auto assembleWeight = [&](mlir::Value qs, mlir::Value hm, int shift,
                              int p) -> mlir::Value {
      mlir::Value shifted =
          (shift == 0) ? qs : u8Imm(vsrlCallee, qs, std::to_string(shift));
      mlir::Value low2 = u8Imm(vandCallee, shifted, "0x03");
      mlir::Value baseI8 = reinterpretToI8(low2); // i8 in [0,3]
      return emitNativeMaskStaticBitBias(
          rewriter, loc, baseI8, hm, /*bitPos=*/p, /*biasWhenBitSet=*/false,
          /*biasImm=*/-4, l8, /*elemSigned=*/true, u8mf2Type, i8mf2Type, vl8,
          opName, role); // i8 in [-4,3]
    };
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    std::string vwmaccVV32Callee = ("__riscv_vwmacc_vv_i32" + l32).str();
    auto vwmaccVV32 = [&](mlir::Value acc, mlir::Value scaleStrip,
                          mlir::Value sumiS) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVV32Callee,
                            mlir::ValueRange{acc, scaleStrip, sumiS, vl8}, opName,
                            role);
    };
    std::string vsextCallee = ("__riscv_vsext_vf2_i16" + l16).str();
    auto liftScaleToI16 = [&](mlir::Value i8scale) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vsextCallee,
                            mlir::ValueRange{i8scale, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t baseByteOff,
                            int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      int64_t totalOff = baseByteOff + laneOff * 2;
      if (totalOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    auto widenF16 = [&](mlir::Value f16strip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                            mlir::ValueRange{f16strip, vl8}, opName, role);
    };
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    auto fmulScalar = [&](mlir::Value vec, mlir::Value scalar) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    auto seedI32 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvI32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
    auto seedI16 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i16m1Type, mvI16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // ===== [loop-order REALIZE] SEL-1 loop-order schedule axis (PURE REALIZE) =====
    // The activation ROW-GROUP loop (nr/4) and the weight COLUMN-GROUP loop (nc/16)
    // are INDEPENDENT -- every out[y,x] is a private K-accumulation -- so either
    // nesting order yields BYTE-IDENTICAL results and an identical hot inner core.
    // Which loop is OUTER is a SCHEDULE axis the caller resolves from the front-door
    // loop_order stamp into `colGroupOuter` (row-group-OUTER == the
    // M1-committed sibling default; col-group-OUTER holds the DRAM-dominant repacked
    // weight panel resident across the row sweep). PURE REALIZE of the SAME loop
    // interchange proven byte-exact for the q4_K min-fold GEMM.
    mlir::LogicalResult status = mlir::success();

    // a = vy + y*nb*activationStride  (the row-group activation base; fn of y ONLY).
    auto emitAGroupBase = [&](mlir::Value y) -> mlir::Value {
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      return rewriter.create<emitc::AddOp>(loc, activationPtrType, activationBase,
                                           aGroupOff);
    };
    // b = vx + x*nb*weightStride  (the repacked weight col-group base; fn of x ONLY).
    // In the col-OUTER nest this is HOISTED above the row sweep (once per col-group).
    auto emitBGroupBase = [&](mlir::Value x) -> mlir::Value {
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      return rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase,
                                           bGroupOff);
    };

    // The per-(y,x) OUTPUT TILE. BYTE-EXACT INVARIANT: emitted IDENTICALLY for both
    // loop orders -- only the two enclosing ForOp headers swap. The row-group-OUTER
    // arm reproduces the M1-committed single fixed nest op-for-op.
    auto emitTile = [&](mlir::Value x, mlir::Value y, mlir::Value bGroup,
                        mlir::Value aGroup) {

        // Activation-column-PASS loop (compile-time, C++).
        for (int64_t cLo = 0; cLo < activationInterleave;
             cLo += columnsPerPass) {
          int64_t cHi = cLo + columnsPerPass;

          std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
          auto seedF32 = [&]() -> mlir::Value {
            return emitOpaqueCallBuilt(
                rewriter, loc, f32m2Type, fmvCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  mlir::Value zero =
                      rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                          .getResult();
                  return {zero, vl8};
                });
          };
          // ===== [S6 strip-outer tiling] weight-strip loop OUTSIDE the block loop.
          // Process one DISJOINT weight strip h at a time, each in its OWN
          // contraction-block loop, so per block loop only THIS strip's accumulators
          // (4 sumf f32m2 + 4 sumi i32m2 + 16 i16 partials) are live -- peak ~36 vreg
          // vs the strip-INTERLEAVED ~64 that spilled (the same q4_K/q6_K register-cliff
          // lever; board objdump on q6_K: 2068 vs1r/vl1r whole-reg spills + 2210 csrr
          // vlenb spill-offset collapsed to ~100). BYTE-EXACT INVARIANT: every out[c][h]
          // accumulates over the SAME blocks in the SAME (l,sh,grp,k,pos,q) order;
          // hoisting h from innermost to outermost only reorders the interleave of
          // INDEPENDENT (disjoint weight-byte / disjoint output) strips, never a single
          // output's fold order. numHalves is the VLEN-derived strip count (2 @VLEN128
          // where the 32-vreg budget forces tiling, 1 @VLEN256 == byte-identical old emit).
          for (int64_t h = 0; h < numHalves; ++h) {
          // sumfVar[c]: per activation column c, THIS strip. Carried across the block
          // loop as an SSA-register VariableOp (never rolled -- the re-roll trap).
          llvm::SmallVector<mlir::Value> sumfVar(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            auto v = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, v, seedF32());
            sumfVar[c] = v;
          }

          // ===== Inner contraction-BLOCK loop: for (l = 0; l < nb; ++l) =====
          auto blockLoop = rewriter.create<emitc::ForOp>(
              loc, sizeLit(0), nb, sizeLit(1), /*bodyBuilder=*/nullptr);
          {
            mlir::OpBuilder::InsertionGuard bg(rewriter);
            rewriter.setInsertionPointToStart(blockLoop.getBody());
            mlir::Value lvar = blockLoop.getInductionVar();

            step("weight_block_base");
            mlir::Value blOff = rewriter.create<emitc::MulOp>(
                loc, sizeType, lvar, sizeLit(weightStride));
            mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                           bGroup, blOff);
            step("act_block_base");
            mlir::Value alOff = rewriter.create<emitc::MulOp>(
                loc, sizeType, lvar, sizeLit(activationStride));
            mlir::Value al = rewriter.create<emitc::AddOp>(
                loc, activationPtrType, aGroup, alOff);

            // -- per-column activation super-block delta d_y_c = *(const float
            // *)&al.d[c]. block_q8_Kx4 stores 4 fp32 deltas at offsets 0,4,8,12.
            llvm::SmallVector<mlir::Value> aD(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c) {
              aD[c] = emitOpaqueCallBuilt(
                  rewriter, loc, floatType, floatReadCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value aDPtr = al;
                    if (c != 0)
                      aDPtr = rewriter.create<emitc::AddOp>(
                          loc, activationPtrType, al, sizeLit(c * 4));
                    mlir::Value aDCast =
                        rewriter.create<emitc::CastOp>(loc, floatPtrConstType,
                                                       aDPtr)
                            .getResult();
                    return {aDCast};
                  },
                  llvm::StringRef("act_scale_scalar"));
            }

            // -- THIS strip's fp16 d strip widened to f32 (per column-lane).
            mlir::Value dF32 = widenF16(loadF16Strip(bl, 0, h * half));

            // ===== Per-column i32 SINGLE accumulator, THIS strip (no min). =====
            llvm::SmallVector<mlir::Value> sumiVar(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c) {
              auto sv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
              sumiVar[c] = sv;
            }

            // ===== Super-half loop. =====
            for (int64_t sh = 0; sh < nSuperHalves; ++sh) {
              for (int64_t grp = 0; grp < 2; ++grp) {
                // SHARED signed scale strips per (shift-quadrant q, sub-half grp):
                // sub-block index sh*8 + q*2 + grp. Reused across columns.
                step("signed_scale_unpack");
                llvm::SmallVector<mlir::Value> scaleVal;
                for (int q = 0; q < 4; ++q) {
                  int64_t sIdx = sh * 8 + q * 2 + grp;
                  mlir::Value s8 = loadI8Strip(
                      bl, weightScalesOffset + sIdx * 16 + h * half);
                  scaleVal.push_back(liftScaleToI16(s8));
                }

                // Per-column i16 partials acc[c][q] over the 16 sub-block positions
                // (THIS strip; the h dimension is the enclosing strip-outer loop;
                // NO 2x8 k-chunk split -- |w| <= 4).
                llvm::SmallVector<llvm::SmallVector<mlir::Value>> acc(
                    activationInterleave);
                if (rolledMainTerm) {
                  // ---- ROLLED whole-K-nest main term ([GAP-EMIT-VSETVL-TAX] /
                  // [K-10] structural GEMM plan): the dominant per-16-position inner
                  // l-loop is materialized as ONE runtime emitc.for so gcc hoists the
                  // e8 vsetvli out of the hot body (collapse the vsetvli storm). The
                  // 4-column x per-strip x per-quadrant i16 partials accVar[c][h][q]
                  // are carried as RESIDENT SSA-register VariableOps (seeded ABOVE the
                  // loop, load-accumulate-store INSIDE it). The runtime position l only
                  // shifts the base pointers: bl+l*16 for the 16-way-interleaved weight
                  // strips (qs AND hmask share the base), al+l*4 for the 4-column-
                  // interleaved q8_Kx4 activation; the compile-time remainder rides the
                  // strip offset. The 3-bit subtractive-hmask weight decode stays INSIDE
                  // the l-loop but OUTSIDE the column loop -> each weight is decoded ONCE
                  // and shared across all 4 activation columns (NO re-decode, NO tile
                  // narrowing -- the whole-K-nest, not a narrow output tile). BYTE-EXACT
                  // to the unrolled emit by construction: the vwmacc16 accumulation order
                  // (l ascending, then column c, then quadrant q, then strip h) is
                  // IDENTICAL -- only the loop is materialized.
                  llvm::SmallVector<llvm::SmallVector<mlir::Value>> accVar(
                      activationInterleave);
                  for (int64_t c = cLo; c < cHi; ++c) {
                    for (int q = 0; q < 4; ++q) {
                      auto v = rewriter.create<emitc::VariableOp>(
                          loc, emitc::LValueType::get(i16m1Type),
                          emitc::OpaqueAttr::get(ctx, ""));
                      rewriter.create<emitc::AssignOp>(loc, v, seedI16());
                      accVar[c].push_back(v);
                    }
                  }
                  auto lLoop = rewriter.create<emitc::ForOp>(
                      loc, sizeLit(0), sizeLit(16), sizeLit(1),
                      /*bodyBuilder=*/nullptr);
                  {
                    mlir::OpBuilder::InsertionGuard ig(rewriter);
                    rewriter.setInsertionPointToStart(lLoop.getBody());
                    mlir::Value lv = lLoop.getInductionVar();
                    mlir::Value lv16 = rewriter.create<emitc::MulOp>(
                        loc, sizeType, lv, sizeLit(16));
                    mlir::Value blL = rewriter.create<emitc::AddOp>(
                        loc, weightPtrType, bl, lv16);
                    mlir::Value lv4 = rewriter.create<emitc::MulOp>(
                        loc, sizeType, lv, sizeLit(4));
                    mlir::Value alL = rewriter.create<emitc::AddOp>(
                        loc, activationPtrType, al, lv4);
                    // THIS strip's weight decode per l: 4 shift-quadrant signed weights,
                    // reused across the interleaved activation columns.
                    llvm::SmallVector<mlir::Value> wq;
                    step("weight_qs_hmask_addr");
                    mlir::Value qs = loadU8Strip(
                        blL, weightQsOffset + (sh * 32 + grp * 16) * 16 + h * half);
                    mlir::Value hm = loadU8Strip(
                        blL, weightHmaskOffset + (grp * 16) * 16 + h * half);
                    for (int q = 0; q < 4; ++q) {
                      int shift = 2 * q;
                      int p = 4 * static_cast<int>(sh) + q;
                      wq.push_back(assembleWeight(qs, hm, shift, p));
                    }
                    for (int64_t c = cLo; c < cHi; ++c) {
                      step("act_quant_addr");
                      // Interleaved q8_Kx4 quant byte for column c, global element
                      // position gpBase = sh*128 + q*32 + grp*16 (the +l rides alL).
                      for (int q = 0; q < 4; ++q) {
                        int64_t gpBase = sh * 128 + q * 32 + grp * 16;
                        mlir::Value aqc =
                            i8Read(alL, activationQuantOffset + gpBase * 4 + c);
                        mlir::Value cur =
                            rewriter
                                .create<emitc::LoadOp>(loc, i16m1Type,
                                                       accVar[c][q])
                                .getResult();
                        rewriter.create<emitc::AssignOp>(
                            loc, accVar[c][q], vwmacc16(cur, aqc, wq[q]));
                      }
                    }
                  }
                  // Load the resident partials back out for the scale fold below.
                  for (int64_t c = cLo; c < cHi; ++c) {
                    for (int q = 0; q < 4; ++q)
                      acc[c].push_back(
                          rewriter
                              .create<emitc::LoadOp>(loc, i16m1Type,
                                                     accVar[c][q])
                              .getResult());
                  }
                } else {
                  // ---- UNROLLED main term (default when the [ROLL] resolver keeps the
                  // full static unroll; the register-resident form). THIS strip only
                  // (the h dimension is the enclosing strip-outer loop). ----
                  for (int64_t c = cLo; c < cHi; ++c) {
                    for (int q = 0; q < 4; ++q)
                      acc[c].push_back(seedI16());
                  }

                  for (int64_t l = 0; l < 16; ++l) {
                    // THIS strip's weight decode per l: 4 shift-quadrant signed weights,
                    // reused across the interleaved activation columns.
                    llvm::SmallVector<mlir::Value> wq;
                    step("weight_qs_hmask_addr");
                    mlir::Value qs = loadU8Strip(
                        bl,
                        weightQsOffset + (sh * 32 + grp * 16 + l) * 16 + h * half);
                    mlir::Value hm = loadU8Strip(
                        bl, weightHmaskOffset + (grp * 16 + l) * 16 + h * half);
                    for (int q = 0; q < 4; ++q) {
                      int shift = 2 * q;
                      int p = 4 * static_cast<int>(sh) + q;
                      wq.push_back(assembleWeight(qs, hm, shift, p));
                    }
                    for (int64_t c = cLo; c < cHi; ++c) {
                      step("act_quant_addr");
                      // Interleaved q8_Kx4 quant byte for column c, global element
                      // position gp = sh*128 + q*32 + grp*16 + l: offset + gp*4 + c.
                      for (int q = 0; q < 4; ++q) {
                        int64_t gp = sh * 128 + q * 32 + grp * 16 + l;
                        mlir::Value aqc =
                            i8Read(al, activationQuantOffset + gp * 4 + c);
                        acc[c][q] = vwmacc16(acc[c][q], aqc, wq[q]);
                      }
                    }
                  }
                }
                // sumi_c += sum over 4 shift-quadrants of signed_scale * acc_c (this strip).
                step("scale_subblock_fold");
                for (int64_t c = cLo; c < cHi; ++c) {
                  mlir::Value cur =
                      rewriter
                          .create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                          .getResult();
                  for (int q = 0; q < 4; ++q)
                    cur = vwmaccVV32(cur, scaleVal[q], acc[c][q]);
                  rewriter.create<emitc::AssignOp>(loc, sumiVar[c], cur);
                }
              }
            }

            // ===== End-of-block SINGLE fold per column, THIS strip (no min). =====
            for (int64_t c = cLo; c < cHi; ++c) {
              mlir::Value d0 = fmulScalar(dF32, aD[c]);
              mlir::Value sumiV =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                      .getResult();
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                      .getResult();
              mlir::Value afterMain = emitOpaqueCallBuilt(
                  rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    return {curF, cvtI32F32(sumiV), d0, vl8};
                  });
              rewriter.create<emitc::AssignOp>(loc, sumfVar[c], afterMain);
            }
          }

          // Per-column store, THIS strip: s + (y*4 + c)*bs + x*16 + h*half.
          std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
          for (int64_t c = cLo; c < cHi; ++c) {
            step("output_addr");
            mlir::Value y4 = rewriter.create<emitc::MulOp>(
                loc, sizeType, y, sizeLit(activationInterleave));
            mlir::Value rowIdx =
                rewriter.create<emitc::AddOp>(loc, sizeType, y4, sizeLit(c));
            mlir::Value rowOff = rewriter.create<emitc::MulOp>(
                loc, sizeType, rowIdx, outputRowStride);
            mlir::Value x16 = rewriter.create<emitc::MulOp>(
                loc, sizeType, x, sizeLit(weightInterleave));
            mlir::Value colOff =
                rewriter.create<emitc::AddOp>(loc, sizeType, rowOff, x16);
            mlir::Value totalOff = colOff;
            if (h * half != 0)
              totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, colOff,
                                                       sizeLit(h * half));
            mlir::Value dst = rewriter.create<emitc::AddOp>(
                loc, floatPtrType, output, totalOff);
            mlir::Value sumfVal =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                    .getResult();
            emitOpaqueCallVoid(rewriter, loc, vseCallee,
                               mlir::ValueRange{dst, sumfVal, vl8}, opName,
                               role);
          }
          } // end [S6 strip-outer] weight-strip loop (h)
        } // end activation-column-PASS loop (cLo)
    };  // end emitTile (byte-identical body for both loop orders)

    if (colGroupOuter) {
      // col-group WEIGHT panel OUTER; row groups sweep INSIDE (weight-resident).
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();
      mlir::Value bGroup = emitBGroupBase(x);
      auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();
      mlir::Value aGroup = emitAGroupBase(y);
      emitTile(x, y, bGroup, aGroup);
    } else {
      // activation ROW panel OUTER; col groups sweep INSIDE (M1-committed default).
      auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();
      mlir::Value aGroup = emitAGroupBase(y);
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();
      mlir::Value bGroup = emitBGroupBase(x);
      emitTile(x, y, bGroup, aGroup);
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    (void)i32m1Type;
    // RESULT-LESS (no monolith token): the typed_repack_gemm_loop_body region is
    // result-less (the repacked lane-wise q3_K GEMM sinks through the output pointer),
    // so unlike the retired direct emitter this body leaf seeds NO dead i32m1 token.
    return mlir::success();
  }

// ===========================================================================
// TERNARY (BitNet-class) 16x1-REPACKED block-as-lane GEVM/GEMM emitters. tq2_0
// and tq1_0 are LINEAR ternary super-blocks: the weight is a TRIT in {-1,0,1},
// the whole 256-element super-block carries ONE fp16 super-block scale, and the
// dot is `sumf += fp16(x.d) * y.d * (sum over 256 of trit * q8)` -- NO
// per-sub-block scale, NO dmin, NO bsums, NO min term. The block-as-lane repack
// erases the per-block cross-lane reduction wall: the 16 interleaved weight
// columns occupy 16 vector lanes and the dot accumulates LANE-WISE via vwmacc.
// The per-super-block integer dot lands in a per-column-strip i16 partial (bound
// well under 32767: a super-half of 128 gives 128*127 = 16256) and is widened
// into ONE i32 accumulator per strip (vwadd_wv), then scaled by the single fp16
// super-block scale into f32. The trit DECODE reuses the SAME ternary-core brick
// formulas as the tq2_0/tq1_0 vec_dot (RVVToEmitCTernaryBinary.cpp): tq2_0's
// 2-bit peel + `-1` bias, tq1_0's base-3 unpack + `-1` bias.
// ===========================================================================

// TERNARY REPACK GEVM front-door body (tq2_0). The byte-exact body of the RETIRED
// monolithic direct emitter emitRepackGemvTQ20Q8K, refactored to take the mapped
// ABI values + block-format facts as PARAMETERS (no monolith-op lookup, no trailing
// unused-result token). It is called ONLY from emitTypedRepackGemvLoopBody's ternary
// branch, gated on the in-region weft_rvv.repack_gemv_ternary_core anti-bypass brick,
// so the ternary repacked GEVM is now CONSTRUCTED through the typed-region front door
// (the q4_0 typed_repack precedent) and byte-exactness to the old direct emitter is by
// construction (the SAME emit code). tq1_0 gets its own base-3 body function.
mlir::LogicalResult VariantToEmitCFunc::emitRepackTernaryGemvBodyTQ20(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t activationQuantOffset,
    int64_t weightInterleave, int64_t half) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*). "mf2"
    // (default) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2
    // (f16 scale m1). "m1" is the RVV0.7.1 whole-LMUL chain i8m1 -> i16m2 -> i32m4.
    llvm::StringRef l8 = coreLmul;                         // mf2 -> mf2; m1 -> m1
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";  // mf2 -> m1;  m1 -> m2
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";  // mf2 -> m2;  m1 -> m4
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // tq2_0 16x1 repack block-format facts: qk 256, weight stride 1056, activation
    // stride 292, weight quant offset 32, activation quant offset 4, interleave 16
    // (passed from the enclosing typed_repack_gemv_loop_body op's pinned attrs).
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-tq2_0 output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // size_t nb = n / QK_K;  size_t nc_groups = nc / 16;
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load / decode helpers ----
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    auto u8Imm = [&](llvm::StringRef callee, mlir::Value v,
                     llvm::StringRef imm) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, imm)
                    .getResult();
            return {v, immV, vl8};
          });
    };
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    // The ternary trit: reinterpret the 2-bit lane (0..3) to i8, then subtract 1
    // (the `-1` ternary bias the q2_K repack LACKS) -> a signed trit in {-1,0,1}.
    std::string vsubI8Callee = ("__riscv_vsub_vx_i8" + l8).str();
    auto decodeTrit = [&](mlir::Value packed, int64_t j) -> mlir::Value {
      mlir::Value shifted = packed;
      if (j != 0)
        shifted = u8Imm(vsrlCallee, packed, std::to_string(2 * j));
      mlir::Value w2 = u8Imm(vandCallee, shifted, "0x03");
      mlir::Value wi8 = emitOpaqueCall(rewriter, loc, i8mf2Type,
                                       reinterpretCallee,
                                       mlir::ValueRange{w2}, opName, role);
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vsubI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value one =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "1")
                    .getResult();
            return {wi8, one, vl8};
          });
    };
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    // vwadd_wv i16 -> i32: the single-scale super-half fold (sumi_i32 += partial_i16
    // lane-wise; NO per-sub-block scale multiply, so a plain widening add suffices).
    std::string vwaddwvCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddwv = [&](mlir::Value acc, mlir::Value partial) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwvCallee,
                            mlir::ValueRange{acc, partial, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      int64_t totalOff = laneOff * 2; // d strip at weight offset 0
      if (totalOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    auto widenF16 = [&](mlir::Value f16strip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                            mlir::ValueRange{f16strip, vl8}, opName, role);
    };
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    auto fmulScalar = [&](mlir::Value vec, mlir::Value scalar) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    auto seedI32 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvI32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
    auto seedI16 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i16m1Type, mvI16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // ===== Outer weight-COLUMN-GROUP loop: for (x = 0; x < nc/16; ++x) =====
    auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();

      // const uint8_t *b = vx + x*nb*1056;
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      mlir::Value bGroup = rewriter.create<emitc::AddOp>(
          loc, weightPtrType, weightBase, bGroupOff);

      // vfloat32 sumf_h = 0 per strip (carried across blocks).
      std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
      auto seedF32 = [&]() -> mlir::Value {
        return emitOpaqueCallBuilt(
            rewriter, loc, f32m2Type, fmvCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                      .getResult();
              return {zero, vl8};
            });
      };
      llvm::SmallVector<mlir::Value> sumfVar;
      for (int64_t h = 0; h < numHalves; ++h) {
        auto v = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(f32m2Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(loc, v, seedF32());
        sumfVar.push_back(v);
      }

      // ===== Inner contraction-BLOCK loop: for (l = 0; l < nb; ++l) =====
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard bg(rewriter);
        rewriter.setInsertionPointToStart(blockLoop.getBody());
        mlir::Value l = blockLoop.getInductionVar();

        // const uint8_t *bl = b + l*1056;   const uint8_t *al = a + l*292;
        step("weight_block_base");
        mlir::Value blOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(weightStride));
        mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                       bGroup, blOff);
        step("act_block_base");
        mlir::Value alOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(activationStride));
        mlir::Value al = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, activationBase, alOff);

        // The activation super-block delta d_y = *(const float *)&al.d (fp32).
        mlir::Value aD = emitOpaqueCallBuilt(
            rewriter, loc, floatType, floatReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, floatPtrConstType, al)
                      .getResult();
              return {aDCast};
            },
            llvm::StringRef("act_scale_scalar"));

        // Per-strip i32 accumulator (single, no min).
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Super-half loop k in {0,1}: accumulate a 128-element i16 partial
        // per strip, then widen-add into the i32 accumulator. =====
        for (int64_t k = 0; k < nSuperHalves; ++k) {
          llvm::SmallVector<mlir::Value> partial;
          for (int64_t h = 0; h < numHalves; ++h)
            partial.push_back(seedI16());
          for (int64_t m = 0; m < 32; ++m) {
            for (int64_t h = 0; h < numHalves; ++h) {
              step("weight_2bit_addr");
              // qs byte index within the super-block = k*32 + m.
              mlir::Value packed = loadU8Strip(
                  bl, weightQuantOffset + (k * 32 + m) * 16 + h * half);
              for (int64_t j = 0; j < 4; ++j) {
                mlir::Value w = decodeTrit(packed, j);
                step("act_quant_addr");
                // activation global position = k*128 + j*32 + m.
                mlir::Value aq = i8Read(
                    al, activationQuantOffset + k * 128 + j * 32 + m);
                partial[h] = vwmacc16(partial[h], aq, w);
              }
            }
          }
          step("superhalf_fold");
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value cur =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                             vwaddwv(cur, partial[h]));
          }
        }

        // ===== End-of-block single-scale fold: sumf += cvt(sumi) * (d_x*d_y). =====
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dStrip = loadF16Strip(bl, h * half);
          mlir::Value d0 = fmulScalar(widenF16(dStrip), aD);
          mlir::Value sumiV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult();
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value after = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {curF, cvtI32F32(sumiV), d0, vl8};
              });
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], after);
        }
      }

      // Per-strip store vse32(s + x*16 + h*half, sumf_h, half).
      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      for (int64_t h = 0; h < numHalves; ++h) {
        step("output_addr");
        mlir::Value x16 = rewriter.create<emitc::MulOp>(
            loc, sizeType, x, sizeLit(weightInterleave));
        mlir::Value totalOff = x16;
        if (h * half != 0)
          totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, x16,
                                                   sizeLit(h * half));
        mlir::Value dst = rewriter.create<emitc::AddOp>(loc, floatPtrType,
                                                        output, totalOff);
        mlir::Value sumfVal =
            rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                .getResult();
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
      }
    }

    // The typed_repack_gemv_loop_body region op is RESULT-LESS (the per-strip
    // lane-wise vse32 is the sink), so unlike the retired monolith direct emitter
    // there is NO trailing unused-result token to seed.
    return mlir::success();
  }

// TERNARY REPACK GEMM (prefill) front-door body (tq2_0). The byte-exact body of the
// RETIRED monolithic direct emitter emitRepackGemmTQ20Q8K, refactored to take the
// mapped ABI values + block-format facts as PARAMETERS (no monolith-op lookup, no
// trailing unused-result token). Called ONLY from emitTypedRepackGemmLoopBody's
// ternary branch, gated on the in-region weft_rvv.repack_gemm_ternary_core
// anti-bypass brick, so byte-exactness to the old direct emitter is by construction.
mlir::LogicalResult VariantToEmitCFunc::emitRepackTernaryGemmBodyTQ20(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQuantOffset,
    int64_t activationQuantOffset, int64_t weightInterleave,
    int64_t activationInterleave, int64_t half, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::StringRef l8 = coreLmul;
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // tq2_0 16x1 repack GEMM block-format facts: qk 256, weight stride 1056,
    // activation stride 1168 (block_q8_Kx4), weight quant offset 32, activation
    // quant offset 16, weight interleave 16, activation interleave 4 (passed from
    // the enclosing typed_repack_gemm_loop_body op's pinned attrs).
    int64_t numHalves = weightInterleave / half;
    int64_t nSuperHalves = qk / 128;
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemm-tq2_0 output not pointer");

    mlir::Value vl8 = sizeLit(half);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("row_group_count");
    mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, rowCount, sizeLit(activationInterleave));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    auto u8Imm = [&](llvm::StringRef callee, mlir::Value v,
                     llvm::StringRef imm) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, imm)
                    .getResult();
            return {v, immV, vl8};
          });
    };
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    std::string vsubI8Callee = ("__riscv_vsub_vx_i8" + l8).str();
    auto decodeTrit = [&](mlir::Value packed, int64_t j) -> mlir::Value {
      mlir::Value shifted = packed;
      if (j != 0)
        shifted = u8Imm(vsrlCallee, packed, std::to_string(2 * j));
      mlir::Value w2 = u8Imm(vandCallee, shifted, "0x03");
      mlir::Value wi8 = emitOpaqueCall(rewriter, loc, i8mf2Type,
                                       reinterpretCallee,
                                       mlir::ValueRange{w2}, opName, role);
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vsubI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value one =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "1")
                    .getResult();
            return {wi8, one, vl8};
          });
    };
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    std::string vwaddwvCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddwv = [&](mlir::Value acc, mlir::Value partial) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwvCallee,
                            mlir::ValueRange{acc, partial, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      int64_t totalOff = laneOff * 2;
      if (totalOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    auto widenF16 = [&](mlir::Value f16strip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                            mlir::ValueRange{f16strip, vl8}, opName, role);
    };
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    auto fmulScalar = [&](mlir::Value vec, mlir::Value scalar) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    auto seedI32 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvI32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
    auto seedI16 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i16m1Type, mvI16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // ===== [loop-order REALIZE] SEL-1 loop-order schedule axis (PURE REALIZE) =====
    // The activation ROW-GROUP loop (nr/4) and the weight COLUMN-GROUP loop (nc/16)
    // are INDEPENDENT -- every out[y,x] is a private K-accumulation -- so either
    // nesting order yields BYTE-IDENTICAL results and an identical hot inner core.
    // Which loop is OUTER is a SCHEDULE axis the caller resolves from the front-door
    // loop_order stamp into `colGroupOuter` (row-group-OUTER == the
    // M1-committed sibling default; col-group-OUTER holds the DRAM-dominant repacked
    // weight panel resident across the row sweep). PURE REALIZE of the SAME loop
    // interchange proven byte-exact for the q4_K min-fold GEMM.

    // a = vy + y*nb*activationStride  (the row-group activation base; fn of y ONLY).
    auto emitAGroupBase = [&](mlir::Value y) -> mlir::Value {
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      return rewriter.create<emitc::AddOp>(loc, activationPtrType, activationBase,
                                           aGroupOff);
    };
    // b = vx + x*nb*weightStride  (the repacked weight col-group base; fn of x ONLY).
    // In the col-OUTER nest this is HOISTED above the row sweep (once per col-group).
    auto emitBGroupBase = [&](mlir::Value x) -> mlir::Value {
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      return rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase,
                                           bGroupOff);
    };

    // The per-(y,x) OUTPUT TILE. BYTE-EXACT INVARIANT: emitted IDENTICALLY for both
    // loop orders -- only the two enclosing ForOp headers swap. The row-group-OUTER
    // arm reproduces the M1-committed single fixed nest op-for-op.
    auto emitTile = [&](mlir::Value x, mlir::Value y, mlir::Value bGroup,
                        mlir::Value aGroup) {

        for (int64_t cLo = 0; cLo < activationInterleave;
             cLo += columnsPerPass) {
        int64_t cHi = cLo + columnsPerPass;

        std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
        auto seedF32 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, fmvCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                        .getResult();
                return {zero, vl8};
              });
        };
        llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumfVar(
            activationInterleave);
        for (int64_t c = cLo; c < cHi; ++c)
          for (int64_t h = 0; h < numHalves; ++h) {
            auto v = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, v, seedF32());
            sumfVar[c].push_back(v);
          }

        // ===== Inner contraction-BLOCK loop: for (l = 0; l < nb; ++l) =====
        auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard bg(rewriter);
          rewriter.setInsertionPointToStart(blockLoop.getBody());
          mlir::Value l = blockLoop.getInductionVar();

          step("weight_block_base");
          mlir::Value blOff = rewriter.create<emitc::MulOp>(
              loc, sizeType, l, sizeLit(weightStride));
          mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                         bGroup, blOff);
          step("act_block_base");
          mlir::Value alOff = rewriter.create<emitc::MulOp>(
              loc, sizeType, l, sizeLit(activationStride));
          mlir::Value al = rewriter.create<emitc::AddOp>(
              loc, activationPtrType, aGroup, alOff);

          // Per-column activation super-block delta d_y_c (4 fp32 at 0,4,8,12).
          llvm::SmallVector<mlir::Value> aD(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            aD[c] = emitOpaqueCallBuilt(
                rewriter, loc, floatType, floatReadCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  mlir::Value aDPtr = al;
                  if (c != 0)
                    aDPtr = rewriter.create<emitc::AddOp>(
                        loc, activationPtrType, al, sizeLit(c * 4));
                  mlir::Value aDCast =
                      rewriter.create<emitc::CastOp>(loc, floatPtrConstType,
                                                     aDPtr)
                          .getResult();
                  return {aDCast};
                },
                llvm::StringRef("act_scale_scalar"));
          }

          // SHARED weight d strip (per-column-lane fp16), widened + reused.
          llvm::SmallVector<mlir::Value> dF32(numHalves);
          for (int64_t h = 0; h < numHalves; ++h)
            dF32[h] = widenF16(loadF16Strip(bl, h * half));

          // Per-column i32 accumulator (single, no min).
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumiVar(
              activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c)
            for (int64_t h = 0; h < numHalves; ++h) {
              auto sv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
              sumiVar[c].push_back(sv);
            }

          // ===== Super-half loop k in {0,1}. =====
          for (int64_t k = 0; k < nSuperHalves; ++k) {
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> partial(
                activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h)
                partial[c].push_back(seedI16());
            for (int64_t m = 0; m < 32; ++m) {
              // SHARED weight trit decode per (m,h,j): reused over columns.
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> wLane(numHalves);
              for (int64_t h = 0; h < numHalves; ++h) {
                step("weight_2bit_addr");
                mlir::Value packed = loadU8Strip(
                    bl, weightQuantOffset + (k * 32 + m) * 16 + h * half);
                for (int64_t j = 0; j < 4; ++j)
                  wLane[h].push_back(decodeTrit(packed, j));
              }
              for (int64_t c = cLo; c < cHi; ++c)
                for (int64_t j = 0; j < 4; ++j) {
                  step("act_quant_addr");
                  mlir::Value aq = i8Read(
                      al, activationQuantOffset +
                              (k * 128 + j * 32 + m) * 4 + c);
                  for (int64_t h = 0; h < numHalves; ++h)
                    partial[c][h] =
                        vwmacc16(partial[c][h], aq, wLane[h][j]);
                }
            }
            step("superhalf_fold");
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value cur =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                        .getResult();
                rewriter.create<emitc::AssignOp>(loc, sumiVar[c][h],
                                                 vwaddwv(cur, partial[c][h]));
              }
          }

          // ===== End-of-block per-column single-scale fold. =====
          for (int64_t c = cLo; c < cHi; ++c)
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value d0 = fmulScalar(dF32[h], aD[c]);
              mlir::Value sumiV =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                      .getResult();
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                      .getResult();
              mlir::Value after = emitOpaqueCallBuilt(
                  rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    return {curF, cvtI32F32(sumiV), d0, vl8};
                  });
              rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], after);
            }
        }

        // Per-column per-strip store: s + (y*4 + c)*bs + x*16 + h*half.
        std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
        for (int64_t c = cLo; c < cHi; ++c)
          for (int64_t h = 0; h < numHalves; ++h) {
            step("output_addr");
            mlir::Value y4 = rewriter.create<emitc::MulOp>(
                loc, sizeType, y, sizeLit(activationInterleave));
            mlir::Value rowIdx =
                rewriter.create<emitc::AddOp>(loc, sizeType, y4, sizeLit(c));
            mlir::Value rowOff = rewriter.create<emitc::MulOp>(
                loc, sizeType, rowIdx, outputRowStride);
            mlir::Value x16 = rewriter.create<emitc::MulOp>(
                loc, sizeType, x, sizeLit(weightInterleave));
            mlir::Value colOff =
                rewriter.create<emitc::AddOp>(loc, sizeType, rowOff, x16);
            mlir::Value totalOff = colOff;
            if (h * half != 0)
              totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, colOff,
                                                       sizeLit(h * half));
            mlir::Value dst = rewriter.create<emitc::AddOp>(
                loc, floatPtrType, output, totalOff);
            mlir::Value sumfVal =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                    .getResult();
            emitOpaqueCallVoid(rewriter, loc, vseCallee,
                               mlir::ValueRange{dst, sumfVal, vl8}, opName,
                               role);
          }
        } // end activation-column-PASS loop (cLo)
    };  // end emitTile (byte-identical body for both loop orders)

    if (colGroupOuter) {
      // col-group WEIGHT panel OUTER; row groups sweep INSIDE (weight-resident).
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();
      mlir::Value bGroup = emitBGroupBase(x);
      auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();
      mlir::Value aGroup = emitAGroupBase(y);
      emitTile(x, y, bGroup, aGroup);
    } else {
      // activation ROW panel OUTER; col groups sweep INSIDE (M1-committed default).
      auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();
      mlir::Value aGroup = emitAGroupBase(y);
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();
      mlir::Value bGroup = emitBGroupBase(x);
      emitTile(x, y, bGroup, aGroup);
    }

    // The typed_repack_gemm_loop_body region op is RESULT-LESS (the per-column
    // per-strip lane-wise vse32 is the sink), so unlike the retired monolith direct
    // emitter there is NO trailing unused-result token to seed.
    return mlir::success();
  }

// TERNARY REPACK GEVM front-door body (tq1_0, BASE-3). The byte-exact body of the
// RETIRED monolithic direct emitter emitRepackGemvTQ10Q8K, refactored to take the
// mapped ABI values + block-format facts (including the base-3 qh SECOND weight-plane
// offset the tq2_0 leaf lacks) as PARAMETERS (no monolith-op lookup, no trailing
// unused-result token). Called ONLY from emitTypedRepackGemvLoopBody's ternary branch,
// gated on the in-region weft_rvv.repack_gemv_ternary_core anti-bypass brick
// (decode_model "tq1_0"), so byte-exactness to the old direct emitter is by
// construction (the SAME emit code).
mlir::LogicalResult VariantToEmitCFunc::emitRepackTernaryGemvBodyTQ10(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t weightQhOffset,
    int64_t activationQuantOffset, int64_t weightInterleave, int64_t half) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::StringRef l8 = coreLmul;
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // tq1_0 16x1 repack block-format facts: qk 256, weight stride 864, activation
    // stride 292, weight qs offset 32, weight qh SECOND-plane offset 800, activation
    // quant offset 4, interleave 16 (passed from the enclosing
    // typed_repack_gemv_loop_body op's pinned attrs + its OPTIONAL weight_qh_byte_offset).
    int64_t numHalves = weightInterleave / half;
    const int64_t pow3[5] = {1, 3, 9, 27, 81};

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-tq1_0 output not pointer");

    mlir::Value vl8 = sizeLit(half);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // ---- BASE-3 trit decode of a loaded u8 strip for digit l (reuses the tq1_0
    // vec_dot ternary-core pipeline: q = (u8)(byte*pow3[l]); xi = (q*3)>>8; xi-1). ----
    std::string vmulU8Callee = ("__riscv_vmul_vx_u8" + l8).str();
    std::string vwmuluCallee = ("__riscv_vwmulu_vx_u16" + l16).str();
    std::string vsrlU16Callee = ("__riscv_vsrl_vx_u16" + l16).str();
    std::string vncvtCallee = ("__riscv_vncvt_x_x_w_u8" + l8).str();
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    std::string vaddI8Callee = ("__riscv_vadd_vx_i8" + l8).str();
    auto decodeBase3 = [&](mlir::Value bytes, int64_t l) -> mlir::Value {
      mlir::Value q = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vmulU8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value p = rewriter.create<emitc::LiteralOp>(
                loc, immI32Type, std::to_string(pow3[l])).getResult();
            return {bytes, p, vl8};
          });
      mlir::Value w = emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vwmuluCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {q, three, vl8};
          });
      mlir::Value xi16 = emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsrlU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value eight =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "8")
                    .getResult();
            return {w, eight, vl8};
          });
      mlir::Value xi8 = emitOpaqueCall(rewriter, loc, u8mf2Type, vncvtCallee,
                                       mlir::ValueRange{xi16, vl8}, opName, role);
      mlir::Value xiI = emitOpaqueCall(rewriter, loc, i8mf2Type,
                                       reinterpretCallee,
                                       mlir::ValueRange{xi8}, opName, role);
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vaddI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value negOne =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "-1")
                    .getResult();
            return {xiI, negOne, vl8};
          });
    };
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    std::string vwaddwvCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddwv = [&](mlir::Value acc, mlir::Value partial) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwvCallee,
                            mlir::ValueRange{acc, partial, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      int64_t totalOff = laneOff * 2;
      if (totalOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    auto widenF16 = [&](mlir::Value f16strip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                            mlir::ValueRange{f16strip, vl8}, opName, role);
    };
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    auto fmulScalar = [&](mlir::Value vec, mlir::Value scalar) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    auto seedI32 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvI32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
    auto seedI16 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i16m1Type, mvI16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // A base-3 region (main qs / tail qs / qh): decode `nDigits` digits of each
    // weight byte in [byteLo, byteHi) at the given weight-plane offset, pairing
    // each with the mapped q8 position via `q8Pos(byteIdx, digit)`, accumulating a
    // fresh i16 partial per strip that is folded into the i32 accumulator at the
    // region end (|partial| <= 160*127 = 20320 << 32767).
    auto emitRegion =
        [&](mlir::Value bl, mlir::Value al, int64_t planeOffset,
            int64_t byteLo, int64_t byteHi, int64_t nDigits,
            llvm::function_ref<int64_t(int64_t, int64_t)> q8Pos,
            llvm::SmallVectorImpl<mlir::Value> &sumiVar) {
          llvm::SmallVector<mlir::Value> partial;
          for (int64_t h = 0; h < numHalves; ++h)
            partial.push_back(seedI16());
          for (int64_t bi = byteLo; bi < byteHi; ++bi) {
            for (int64_t h = 0; h < numHalves; ++h) {
              step("weight_base3_addr");
              mlir::Value bytes =
                  loadU8Strip(bl, planeOffset + bi * 16 + h * half);
              for (int64_t l = 0; l < nDigits; ++l) {
                mlir::Value w = decodeBase3(bytes, l);
                step("act_quant_addr");
                mlir::Value aq =
                    i8Read(al, activationQuantOffset + q8Pos(bi, l));
                partial[h] = vwmacc16(partial[h], aq, w);
              }
            }
          }
          step("region_fold");
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value cur =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                             vwaddwv(cur, partial[h]));
          }
        };

    // ===== Outer weight-COLUMN-GROUP loop: for (x = 0; x < nc/16; ++x) =====
    auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();

      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      mlir::Value bGroup = rewriter.create<emitc::AddOp>(
          loc, weightPtrType, weightBase, bGroupOff);

      std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
      auto seedF32 = [&]() -> mlir::Value {
        return emitOpaqueCallBuilt(
            rewriter, loc, f32m2Type, fmvCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                      .getResult();
              return {zero, vl8};
            });
      };
      llvm::SmallVector<mlir::Value> sumfVar;
      for (int64_t h = 0; h < numHalves; ++h) {
        auto v = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(f32m2Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(loc, v, seedF32());
        sumfVar.push_back(v);
      }

      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard bg(rewriter);
        rewriter.setInsertionPointToStart(blockLoop.getBody());
        mlir::Value l = blockLoop.getInductionVar();

        step("weight_block_base");
        mlir::Value blOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(weightStride));
        mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                       bGroup, blOff);
        step("act_block_base");
        mlir::Value alOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(activationStride));
        mlir::Value al = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, activationBase, alOff);

        mlir::Value aD = emitOpaqueCallBuilt(
            rewriter, loc, floatType, floatReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, floatPtrConstType, al)
                      .getResult();
              return {aDCast};
            },
            llvm::StringRef("act_scale_scalar"));

        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // (a) main qs: bytes 0..31, 5 digits, q8 pos = l*32 + bi.
        emitRegion(bl, al, weightQuantOffset, 0, 32, 5,
                   [](int64_t bi, int64_t l) { return l * 32 + bi; }, sumiVar);
        // (b) tail qs: bytes 32..47, 5 digits, q8 pos = 160 + l*16 + (bi-32).
        emitRegion(bl, al, weightQuantOffset, 32, 48, 5,
                   [](int64_t bi, int64_t l) {
                     return 160 + l * 16 + (bi - 32);
                   },
                   sumiVar);
        // (c) qh: bytes 0..3, 4 digits, q8 pos = 240 + l*4 + bi.
        emitRegion(bl, al, weightQhOffset, 0, 4, 4,
                   [](int64_t bi, int64_t l) { return 240 + l * 4 + bi; },
                   sumiVar);

        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dStrip = loadF16Strip(bl, h * half);
          mlir::Value d0 = fmulScalar(widenF16(dStrip), aD);
          mlir::Value sumiV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult();
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value after = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {curF, cvtI32F32(sumiV), d0, vl8};
              });
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], after);
        }
      }

      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      for (int64_t h = 0; h < numHalves; ++h) {
        step("output_addr");
        mlir::Value x16 = rewriter.create<emitc::MulOp>(
            loc, sizeType, x, sizeLit(weightInterleave));
        mlir::Value totalOff = x16;
        if (h * half != 0)
          totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, x16,
                                                   sizeLit(h * half));
        mlir::Value dst = rewriter.create<emitc::AddOp>(loc, floatPtrType,
                                                        output, totalOff);
        mlir::Value sumfVal =
            rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                .getResult();
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
      }
    }

    // The typed_repack_gemv_loop_body region op is RESULT-LESS (the per-strip
    // lane-wise vse32 is the sink), so unlike the retired monolith direct emitter
    // there is NO trailing unused-result token to seed.
    return mlir::success();
  }

// TERNARY REPACK GEMM (prefill) front-door body (tq1_0, BASE-3). The byte-exact body
// of the RETIRED monolithic direct emitter emitRepackGemmTQ10Q8K, refactored to take
// the mapped ABI values + block-format facts (including the base-3 qh SECOND
// weight-plane offset) as PARAMETERS (no monolith-op lookup, no trailing unused-result
// token). Called ONLY from emitTypedRepackGemmLoopBody's ternary branch, gated on the
// in-region weft_rvv.repack_gemm_ternary_core anti-bypass brick (decode_model
// "tq1_0"), so byte-exactness to the old direct emitter is by construction.
mlir::LogicalResult VariantToEmitCFunc::emitRepackTernaryGemmBodyTQ10(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQuantOffset,
    int64_t weightQhOffset, int64_t activationQuantOffset,
    int64_t weightInterleave, int64_t activationInterleave, int64_t half, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::StringRef l8 = coreLmul;
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // tq1_0 16x1 repack GEMM block-format facts: qk 256, weight stride 864,
    // activation stride 1168 (block_q8_Kx4), weight qs offset 32, weight qh
    // SECOND-plane offset 800, activation quant offset 16, weight interleave 16,
    // activation interleave 4 (passed from the enclosing typed_repack_gemm_loop_body
    // op's pinned attrs + its OPTIONAL weight_qh_byte_offset).
    int64_t numHalves = weightInterleave / half;
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave;
    const int64_t pow3[5] = {1, 3, 9, 27, 81};

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemm-tq1_0 output not pointer");

    mlir::Value vl8 = sizeLit(half);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("row_group_count");
    mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, rowCount, sizeLit(activationInterleave));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    std::string vmulU8Callee = ("__riscv_vmul_vx_u8" + l8).str();
    std::string vwmuluCallee = ("__riscv_vwmulu_vx_u16" + l16).str();
    std::string vsrlU16Callee = ("__riscv_vsrl_vx_u16" + l16).str();
    std::string vncvtCallee = ("__riscv_vncvt_x_x_w_u8" + l8).str();
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    std::string vaddI8Callee = ("__riscv_vadd_vx_i8" + l8).str();
    auto decodeBase3 = [&](mlir::Value bytes, int64_t l) -> mlir::Value {
      mlir::Value q = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vmulU8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value p = rewriter.create<emitc::LiteralOp>(
                loc, immI32Type, std::to_string(pow3[l])).getResult();
            return {bytes, p, vl8};
          });
      mlir::Value w = emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vwmuluCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {q, three, vl8};
          });
      mlir::Value xi16 = emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsrlU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value eight =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "8")
                    .getResult();
            return {w, eight, vl8};
          });
      mlir::Value xi8 = emitOpaqueCall(rewriter, loc, u8mf2Type, vncvtCallee,
                                       mlir::ValueRange{xi16, vl8}, opName, role);
      mlir::Value xiI = emitOpaqueCall(rewriter, loc, i8mf2Type,
                                       reinterpretCallee,
                                       mlir::ValueRange{xi8}, opName, role);
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vaddI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location) -> llvm::SmallVector<mlir::Value> {
            mlir::Value negOne =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "-1")
                    .getResult();
            return {xiI, negOne, vl8};
          });
    };
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    std::string vwaddwvCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddwv = [&](mlir::Value acc, mlir::Value partial) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwvCallee,
                            mlir::ValueRange{acc, partial, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      int64_t totalOff = laneOff * 2;
      if (totalOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    auto widenF16 = [&](mlir::Value f16strip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                            mlir::ValueRange{f16strip, vl8}, opName, role);
    };
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    auto fmulScalar = [&](mlir::Value vec, mlir::Value scalar) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    auto seedI32 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvI32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
    auto seedI16 = [&]() -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i16m1Type, mvI16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // A base-3 region with the weight trit decode SHARED across the activation
    // columns [cLo, cHi): decode each digit's trit strip once, then vwmacc it into
    // every column's i16 partial with that column's own interleaved q8 read.
    auto emitRegion =
        [&](mlir::Value bl, mlir::Value al, int64_t planeOffset,
            int64_t byteLo, int64_t byteHi, int64_t nDigits, int64_t cLo,
            int64_t cHi,
            llvm::function_ref<int64_t(int64_t, int64_t)> q8Pos,
            llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value>> &sumiVar) {
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> partial(
              activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c)
            for (int64_t h = 0; h < numHalves; ++h)
              partial[c].push_back(seedI16());
          for (int64_t bi = byteLo; bi < byteHi; ++bi) {
            // SHARED trit decode per (h, digit): wLane[h][l], reused over columns.
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> wLane(numHalves);
            for (int64_t h = 0; h < numHalves; ++h) {
              step("weight_base3_addr");
              mlir::Value bytes =
                  loadU8Strip(bl, planeOffset + bi * 16 + h * half);
              for (int64_t l = 0; l < nDigits; ++l)
                wLane[h].push_back(decodeBase3(bytes, l));
            }
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t l = 0; l < nDigits; ++l) {
                step("act_quant_addr");
                mlir::Value aq = i8Read(
                    al, activationQuantOffset + q8Pos(bi, l) * 4 + c);
                for (int64_t h = 0; h < numHalves; ++h)
                  partial[c][h] = vwmacc16(partial[c][h], aq, wLane[h][l]);
              }
          }
          step("region_fold");
          for (int64_t c = cLo; c < cHi; ++c)
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value cur =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                      .getResult();
              rewriter.create<emitc::AssignOp>(loc, sumiVar[c][h],
                                               vwaddwv(cur, partial[c][h]));
            }
        };

    // ===== [loop-order REALIZE] SEL-1 loop-order schedule axis (PURE REALIZE) =====
    // The activation ROW-GROUP loop (nr/4) and the weight COLUMN-GROUP loop (nc/16)
    // are INDEPENDENT -- every out[y,x] is a private K-accumulation -- so either
    // nesting order yields BYTE-IDENTICAL results and an identical hot inner core.
    // Which loop is OUTER is a SCHEDULE axis the caller resolves from the front-door
    // loop_order stamp into `colGroupOuter` (row-group-OUTER == the
    // M1-committed sibling default; col-group-OUTER holds the DRAM-dominant repacked
    // weight panel resident across the row sweep). PURE REALIZE of the SAME loop
    // interchange proven byte-exact for the q4_K min-fold GEMM.

    // a = vy + y*nb*activationStride  (the row-group activation base; fn of y ONLY).
    auto emitAGroupBase = [&](mlir::Value y) -> mlir::Value {
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      return rewriter.create<emitc::AddOp>(loc, activationPtrType, activationBase,
                                           aGroupOff);
    };
    // b = vx + x*nb*weightStride  (the repacked weight col-group base; fn of x ONLY).
    // In the col-OUTER nest this is HOISTED above the row sweep (once per col-group).
    auto emitBGroupBase = [&](mlir::Value x) -> mlir::Value {
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      return rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase,
                                           bGroupOff);
    };

    // The per-(y,x) OUTPUT TILE. BYTE-EXACT INVARIANT: emitted IDENTICALLY for both
    // loop orders -- only the two enclosing ForOp headers swap. The row-group-OUTER
    // arm reproduces the M1-committed single fixed nest op-for-op.
    auto emitTile = [&](mlir::Value x, mlir::Value y, mlir::Value bGroup,
                        mlir::Value aGroup) {

        for (int64_t cLo = 0; cLo < activationInterleave;
             cLo += columnsPerPass) {
        int64_t cHi = cLo + columnsPerPass;

        std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
        auto seedF32 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, fmvCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                        .getResult();
                return {zero, vl8};
              });
        };
        llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumfVar(
            activationInterleave);
        for (int64_t c = cLo; c < cHi; ++c)
          for (int64_t h = 0; h < numHalves; ++h) {
            auto v = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, v, seedF32());
            sumfVar[c].push_back(v);
          }

        auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard bg(rewriter);
          rewriter.setInsertionPointToStart(blockLoop.getBody());
          mlir::Value l = blockLoop.getInductionVar();

          step("weight_block_base");
          mlir::Value blOff = rewriter.create<emitc::MulOp>(
              loc, sizeType, l, sizeLit(weightStride));
          mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                         bGroup, blOff);
          step("act_block_base");
          mlir::Value alOff = rewriter.create<emitc::MulOp>(
              loc, sizeType, l, sizeLit(activationStride));
          mlir::Value al = rewriter.create<emitc::AddOp>(
              loc, activationPtrType, aGroup, alOff);

          llvm::SmallVector<mlir::Value> aD(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            aD[c] = emitOpaqueCallBuilt(
                rewriter, loc, floatType, floatReadCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  mlir::Value aDPtr = al;
                  if (c != 0)
                    aDPtr = rewriter.create<emitc::AddOp>(
                        loc, activationPtrType, al, sizeLit(c * 4));
                  mlir::Value aDCast =
                      rewriter.create<emitc::CastOp>(loc, floatPtrConstType,
                                                     aDPtr)
                          .getResult();
                  return {aDCast};
                },
                llvm::StringRef("act_scale_scalar"));
          }

          llvm::SmallVector<mlir::Value> dF32(numHalves);
          for (int64_t h = 0; h < numHalves; ++h)
            dF32[h] = widenF16(loadF16Strip(bl, h * half));

          llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumiVar(
              activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c)
            for (int64_t h = 0; h < numHalves; ++h) {
              auto sv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
              sumiVar[c].push_back(sv);
            }

          emitRegion(bl, al, weightQuantOffset, 0, 32, 5, cLo, cHi,
                     [](int64_t bi, int64_t l) { return l * 32 + bi; },
                     sumiVar);
          emitRegion(bl, al, weightQuantOffset, 32, 48, 5, cLo, cHi,
                     [](int64_t bi, int64_t l) {
                       return 160 + l * 16 + (bi - 32);
                     },
                     sumiVar);
          emitRegion(bl, al, weightQhOffset, 0, 4, 4, cLo, cHi,
                     [](int64_t bi, int64_t l) { return 240 + l * 4 + bi; },
                     sumiVar);

          for (int64_t c = cLo; c < cHi; ++c)
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value d0 = fmulScalar(dF32[h], aD[c]);
              mlir::Value sumiV =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                      .getResult();
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                      .getResult();
              mlir::Value after = emitOpaqueCallBuilt(
                  rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    return {curF, cvtI32F32(sumiV), d0, vl8};
                  });
              rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], after);
            }
        }

        std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
        for (int64_t c = cLo; c < cHi; ++c)
          for (int64_t h = 0; h < numHalves; ++h) {
            step("output_addr");
            mlir::Value y4 = rewriter.create<emitc::MulOp>(
                loc, sizeType, y, sizeLit(activationInterleave));
            mlir::Value rowIdx =
                rewriter.create<emitc::AddOp>(loc, sizeType, y4, sizeLit(c));
            mlir::Value rowOff = rewriter.create<emitc::MulOp>(
                loc, sizeType, rowIdx, outputRowStride);
            mlir::Value x16 = rewriter.create<emitc::MulOp>(
                loc, sizeType, x, sizeLit(weightInterleave));
            mlir::Value colOff =
                rewriter.create<emitc::AddOp>(loc, sizeType, rowOff, x16);
            mlir::Value totalOff = colOff;
            if (h * half != 0)
              totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, colOff,
                                                       sizeLit(h * half));
            mlir::Value dst = rewriter.create<emitc::AddOp>(
                loc, floatPtrType, output, totalOff);
            mlir::Value sumfVal =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                    .getResult();
            emitOpaqueCallVoid(rewriter, loc, vseCallee,
                               mlir::ValueRange{dst, sumfVal, vl8}, opName,
                               role);
          }
        } // end activation-column-PASS loop (cLo)
    };  // end emitTile (byte-identical body for both loop orders)

    if (colGroupOuter) {
      // col-group WEIGHT panel OUTER; row groups sweep INSIDE (weight-resident).
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();
      mlir::Value bGroup = emitBGroupBase(x);
      auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();
      mlir::Value aGroup = emitAGroupBase(y);
      emitTile(x, y, bGroup, aGroup);
    } else {
      // activation ROW panel OUTER; col groups sweep INSIDE (M1-committed default).
      auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();
      mlir::Value aGroup = emitAGroupBase(y);
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard cg(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      mlir::Value x = colLoop.getInductionVar();
      mlir::Value bGroup = emitBGroupBase(x);
      emitTile(x, y, bGroup, aGroup);
    }

    // The typed_repack_gemm_loop_body region op is RESULT-LESS (the per-column
    // per-strip lane-wise vse32 is the sink), so unlike the retired monolith direct
    // emitter there is NO trailing unused-result token to seed.
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
