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

// Typed symmetric no-min K-quant repack artifact consumers.

// q6_K x q8_K 16x1-REPACKED GEVM (decode) emitter. q6_K is the LARGEST K-quant
// structural delta over q4_K/q5_K: the 6-bit weight VALUE is assembled LANE-WISE
// from a low-4-bit ql plane + a high-2-bit qh plane --
// `((ql & 0xF) | (((qh >> shift) & 3) << 4)) - 32` (a SIGNED value in [-32,31]) --
// weighted by 16 SIGNED int8 per-16-element sub-block scales, and folded through a
// SINGLE per-strip fp accumulator. There is NO dmin, NO per-sub-block min, NO bsums:
// the -32 offset-binary bias lives INSIDE each weight lane (vsub_vx_i8 by 32), and
// the scale is SIGN-extended (vsext, NOT the q4_K zero-extend). Reuses the q4_K/q5_K
// K-quant block-as-lane scaffold (LANE-WISE vwmacc dot, VLEN-derived 8/16-lane
// strips, per-strip i32->f32 fold); the 6-bit two-plane assembly + signed-scale +
// single-accumulator no-min fold are the q6_K-specific new pieces. NUMERIC STATUS:
// oracle-verified vs an INDEPENDENT scalar q6_K dequant-matmul reference (from the
// ORIGINAL pre-repack q6_K; controls QH-off / SCALE-perturb / no-min structure).
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemvBodyQ6K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQlOffset, int64_t activationQuantOffset,
    int64_t weightScalesOffset, int64_t weightQhOffset, int64_t nSubblocks,
    int64_t weightInterleave, int64_t half, bool rolledMainTerm) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the 16-way
    // interleaved repack reads the SAME bytes either way). "mf2" (default, absent)
    // is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2 (f16 scale m1),
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
    // The 6-bit weight lane is a SIGNED i8 in [-32,31] (offset-binary, -32 baked in);
    // the ql|qh assembly runs on the UNSIGNED u8 raw value first, then vsub 32.
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

    // The 16x1 repacked q6_K (K-quant super-block) GEVM block-format structural
    // facts (I4 mirror, pinned by the verifier): QK_K=256, block_q6_Kx16 weight
    // stride 3360 (16 fp16 d + 256 signed int8 scales + 1024 qh high-2-bit + 2048 ql
    // low-4-bit), block_q8_K activation stride 292 (fp32 d + 256 int8 quants + 16
    // int16 bsums, bsums UNUSED), ql at +1312, qh at +288, signed scales at +32,
    // activation int8 quants at +4, 16 weight columns per group, 16 sub-blocks of 16,
    // and the VLEN-derived e8 half width.
    // qk/weightStride/activationStride/weightQlOffset/activationQuantOffset/
    // weightScalesOffset/weightQhOffset/nSubblocks/weightInterleave/half are
    // PARAMETERS now (the loop body op's pinned attrs, read by the K-quant no-min
    // branch of emitTypedRepackGemvLoopBody and passed in).
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 8 sub-blocks (of 16) / super-half
    (void)subPerSuper;
    // q6_K super-half quadrants: within a 128-element super-half the 4 disjoint
    // 32-element quadrants (element bases +0/+32/+64/+96) draw from the SAME ql byte
    // (low nibble -> quad 0/1, high nibble -> quad 2/3) split across TWO ql streams
    // (quad 0/2 read ql[l], quad 1/3 read ql[l+32]) and the SAME qh byte at 2-bit
    // shifts 0/2/4/6. Each quadrant is 2 sub-blocks of 16 (scale changes at l==16).
    struct QuadInfo {
      int qlStream; // 0 -> ql[l];  1 -> ql[l+32]
      int highNib;  // 0 -> ql & 0xF; 1 -> ql >> 4
      int qhShift;  // 0/2/4/6
    };
    const QuadInfo quads[4] = {
        {0, 0, 0}, // quad 0: ql[l] low  nibble | (qh>>0 & 3)<<4  -> elems +0
        {1, 0, 2}, // quad 1: ql[l+32] low  nibble | (qh>>2 & 3)<<4 -> elems +32
        {0, 1, 4}, // quad 2: ql[l] high nibble | (qh>>4 & 3)<<4  -> elems +64
        {1, 1, 6}, // quad 3: ql[l+32] high nibble | (qh>>6 & 3)<<4 -> elems +96
    };

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-q6_K output not pointer");
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
    // UNSIGNED u8 contiguous strip load (the ql/qh planes).
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
    // vand_vx_u8 / vsrl_vx_u8 / vsll_vx_u8 -- the 6-bit ql|qh assembly bit-dance.
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    std::string vsllCallee = ("__riscv_vsll_vx_u8" + l8).str();
    std::string vorCallee = ("__riscv_vor_vv_u8" + l8).str();
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
    auto u8Or = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, u8mf2Type, vorCallee,
                            mlir::ValueRange{a, b, vl8}, opName, role);
    };
    // qh 2-bit quadrant plane: ((qh >> shift) & 3) << 4  (a mask q4_K/q5_K NEVER
    // emit -- q5_K masks ONE bit 0x01, q6_K masks TWO bits 0x03).
    auto qhQuadBits = [&](mlir::Value qh, int shift) -> mlir::Value {
      mlir::Value sel =
          (shift == 0) ? qh : u8Imm(vsrlCallee, qh, std::to_string(shift));
      return u8Imm(vsllCallee, u8Imm(vandCallee, sel, "0x03"), "4");
    };
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    // vsub_vx_i8: the q6_K -32 offset-binary bias, folded LANE-WISE straight into
    // each weight lane (NO separate min term). raw u8 [0,63] -> signed i8 [-32,31].
    std::string vsubI8Callee = ("__riscv_vsub_vx_i8" + l8).str();
    auto subBias = [&](mlir::Value i8vec) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vsubI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "32")
                    .getResult();
            return {i8vec, immV, vl8};
          });
    };
    // Assemble ONE 6-bit signed weight strip from a ql strip + the qh strip.
    auto assembleWeight = [&](mlir::Value ql, mlir::Value qh, int highNib,
                              int qhShift) -> mlir::Value {
      mlir::Value nib = highNib ? u8Imm(vsrlCallee, ql, "4")
                                : u8Imm(vandCallee, ql, "0x0F");
      mlir::Value raw = u8Or(nib, qhQuadBits(qh, qhShift)); // u8 in [0,63]
      return subBias(reinterpretToI8(raw));                 // i8 in [-32,31]
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
    // vsext_vf2 i8 strip -> i16 strip: the SIGNED per-sub-block scale (NOT the q4_K
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

      // const uint8_t *b = vx + x*nb*3360;  (the q6_Kx16 column group base).
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

        // const uint8_t *bl = b + l*3360;   const uint8_t *al = a + l*292;
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
        // stores a FLOAT (4 bytes) at offset 0. q6_K has a SINGLE d scale (no dmin),
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

        // ===== Per-block i32 accumulator per strip (the SINGLE q6_K accumulator;
        // NO bsums/min accumulator). =====
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Super-half loop: for (j = 0; j < QK_K/128; ++j) =====
        // Each super-half is 4 quadrants of 32 = 8 sub-blocks of 16. For each
        // (quadrant, sub-half of 16) load the SIGNED int8 scale strip, then run the
        // integer dot in 2 chunks of 8 (i16 guard) and fold scale-weighted to i32.
        for (int64_t j = 0; j < nSuperHalves; ++j) {
          for (int64_t sh = 0; sh < 2; ++sh) {
            // Load the 4 quadrant SIGNED scale strips (reused across k=0,1). The
            // sub-block index for (quad q, sub-half sh) is j*8 + q*2 + sh.
            step("signed_scale_unpack");
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> scaleVal(numHalves);
            for (int64_t h = 0; h < numHalves; ++h) {
              for (int q = 0; q < 4; ++q) {
                int64_t sIdx = j * 8 + q * 2 + sh;
                mlir::Value s8 = loadI8Strip(
                    bl, weightScalesOffset + sIdx * 16 + h * half);
                scaleVal[h].push_back(liftScaleToI16(s8));
              }
            }

            for (int64_t k = 0; k < 2; ++k) {
              // i16 partials: acc[h][q] over the 8 positions in this chunk.
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> acc(numHalves);
              if (rolledMainTerm) {
                // ---- ROLLED whole-K-nest main term ([GAP-EMIT-KQUANT-GEVM-TILE-
                // ROUNDTRIP] maturity lever): the dominant per-8-position inner p-loop is
                // emitted as ONE runtime emitc.for; the per-strip x per-quadrant i16
                // partials acc[h][q] are carried as RESIDENT SSA-register VariableOps
                // (seeded ABOVE the loop, load-accumulate-store INSIDE it). The runtime
                // position p only shifts the weight/activation base pointers (bl+p*16,
                // al+p); the compile-time remainder rides the strip-load helper. BYTE-EXACT
                // to the unrolled emit by construction: the vwmacc16 integer accumulation
                // order (p ascending, then h, then quadrant q) is IDENTICAL -- only the loop
                // is materialized instead of unrolled.
                llvm::SmallVector<llvm::SmallVector<mlir::Value>> accVar(numHalves);
                for (int64_t h = 0; h < numHalves; ++h)
                  for (int q = 0; q < 4; ++q) {
                    auto v = rewriter.create<emitc::VariableOp>(
                        loc, emitc::LValueType::get(i16m1Type),
                        emitc::OpaqueAttr::get(ctx, ""));
                    rewriter.create<emitc::AssignOp>(loc, v, seedI16());
                    accVar[h].push_back(v);
                  }
                auto pLoop = rewriter.create<emitc::ForOp>(
                    loc, sizeLit(0), sizeLit(8), sizeLit(1), /*bodyBuilder=*/nullptr);
                {
                  mlir::OpBuilder::InsertionGuard ig(rewriter);
                  rewriter.setInsertionPointToStart(pLoop.getBody());
                  mlir::Value pv = pLoop.getInductionVar();
                  int64_t llBase = sh * 16 + k * 8; // ll = llBase + p
                  // Weight strips advance 16 bytes / position (16-way interleave), the
                  // scalar activation 1 byte / position: shift the base pointers by p.
                  mlir::Value pv16 = rewriter.create<emitc::MulOp>(
                      loc, sizeType, pv, sizeLit(16));
                  mlir::Value blP = rewriter.create<emitc::AddOp>(
                      loc, weightPtrType, bl, pv16);
                  mlir::Value alP = rewriter.create<emitc::AddOp>(
                      loc, activationPtrType, al, pv);
                  step("act_quant_addr");
                  mlir::Value aQ[4];
                  for (int q = 0; q < 4; ++q)
                    aQ[q] = i8Read(alP, activationQuantOffset + j * 128 +
                                            q * 32 + llBase);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    step("weight_ql_qh_addr");
                    mlir::Value qlA = loadU8Strip(
                        blP, weightQlOffset + (j * 64 + llBase) * 16 + h * half);
                    mlir::Value qlB = loadU8Strip(
                        blP, weightQlOffset + (j * 64 + 32 + llBase) * 16 + h * half);
                    mlir::Value qh = loadU8Strip(
                        blP, weightQhOffset + (j * 32 + llBase) * 16 + h * half);
                    for (int q = 0; q < 4; ++q) {
                      mlir::Value qlSel = (quads[q].qlStream == 0) ? qlA : qlB;
                      mlir::Value w = assembleWeight(qlSel, qh, quads[q].highNib,
                                                     quads[q].qhShift);
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

                for (int64_t p = 0; p < 8; ++p) {
                  int64_t ll = sh * 16 + k * 8 + p; // position within quadrant (0..31)
                  // 4 quadrant activations (scalar, shared across strips): global
                  // element position j*128 + q*32 + ll.
                  step("act_quant_addr");
                  mlir::Value aQ[4];
                  for (int q = 0; q < 4; ++q)
                    aQ[q] = i8Read(al, activationQuantOffset + j * 128 +
                                           q * 32 + ll);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    step("weight_ql_qh_addr");
                    // ql[l] stream (quads 0/2) + ql[l+32] stream (quads 1/3) + qh.
                    mlir::Value qlA = loadU8Strip(
                        bl, weightQlOffset + (j * 64 + ll) * 16 + h * half);
                    mlir::Value qlB = loadU8Strip(
                        bl, weightQlOffset + (j * 64 + 32 + ll) * 16 + h * half);
                    mlir::Value qh = loadU8Strip(
                        bl, weightQhOffset + (j * 32 + ll) * 16 + h * half);
                    for (int q = 0; q < 4; ++q) {
                      mlir::Value qlSel = (quads[q].qlStream == 0) ? qlA : qlB;
                      mlir::Value w = assembleWeight(qlSel, qh, quads[q].highNib,
                                                     quads[q].qhShift);
                      acc[h][q] = vwmacc16(acc[h][q], aQ[q], w);
                    }
                  }
                }
              }
              // sumi += sum over 4 quadrants of signed_scale * acc (i16->i32).
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
        }

        // ===== End-of-block SINGLE fold per strip: sumf += cvt(sumi) * (d_x*d_y).
        // NO min term (no vfnmsac): q6_K's -32 offset-binary is inside each weight. =
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
    // result-less (the repacked lane-wise q6_K GEVM sinks through the output
    // pointer, not an SSA vector), so unlike the retired direct emitter this body
    // leaf seeds NO dead i32m1 result token.
    return mlir::success();
  }

// q6_K x q8_K 16x1-REPACKED PREFILL GEMM emitter (S6 STRIP-OUTER TILED, front-door body
// leaf). [G8 stage-3, board-remeasured 2026-07-14] The weight-STRIP loop (numHalves) is
// hoisted OUTSIDE the contraction-block loop so only THIS strip's accumulators are live
// per block loop (~36 vreg vs the strip-interleaved ~64 that spilled). The earlier
// "output tiling board-proven NULL on q6_K" note was FALSIFIED: the ~2068 whole-reg
// spill (vs1r/vl1r) + 2210 csrr(vlenb) it blamed on "two-plane 6-bit reconstruction"
// was in fact the ACCUMULATOR register-cliff that strip-outer scoping removes (objdump
// total 20960->9215, spill ~22x fewer; MACs identical). Byte-exact to the old strip-
// inner emit (27/27 shapes, DYNAMIC per-bit) => also byte-identical numerics to the
// retired direct emitter emitRepackGemmQ6KQ8K. Board: q6_K prefill cold ratio (vs vl128
// block-dot, clang-18 sym) 0.15->0.96(nr16)/1.14(nr64) WIN, HOT ~1.11-1.20.
// The q6_K prefill sibling of emitRepackKQuantGemvBodyQ6K: the SAME 6-bit ql|qh
// two-plane signed-weight assembly (-32
// bias), the SAME 16 signed int8 scales, and the SAME SINGLE-accumulator no-min
// fold, with the weight decode AMORTIZED once per 16-weight group across the 4 (and,
// across the row loop, M) interleaved block_q8_Kx4 activation columns. Oracle-verified
// vs an INDEPENDENT scalar q6_K dequant-matmul reference (controls QH-off /
// SCALE-perturb / ROWROT / no-min).
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemmBodyQ6K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQlOffset,
    int64_t activationQuantOffset, int64_t weightScalesOffset,
    int64_t weightQhOffset, int64_t nSubblocks, int64_t weightInterleave,
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

    // The 16x1 repacked q6_K PREFILL GEMM block-format facts (I4 mirror, verifier-
    // pinned): QK_K=256, block_q6_Kx16 weight stride 3360 (SAME weight ABI as the
    // GEVM), block_q8_Kx4 activation stride 1168 (4 fp32 d + 1024 int8 + 64 int16
    // bsums, bsums UNUSED), ql at +1312, qh at +288, signed scales at +32,
    // interleaved activation quants at +16, 16 weight columns / 4 activation columns,
    // 16 sub-blocks of 16, VLEN-derived e8 half width.
    // qk/weightStride/activationStride/weightQlOffset/activationQuantOffset/
    // weightScalesOffset/weightQhOffset/nSubblocks/weightInterleave/
    // activationInterleave/half are PARAMETERS now (the loop body op's pinned attrs,
    // read by the K-quant no-min branch of emitTypedRepackGemmLoopBody and passed in).
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 8
    (void)subPerSuper;
    // RVV1.0 (fractional chain) folds all 4 activation columns per pass; RVV0.7.1
    // (whole-LMUL) folds ONE column per pass to keep the per-pass live set bounded.
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave; // 1 @rvv07; 4 @rvv1.0

    struct QuadInfo {
      int qlStream;
      int highNib;
      int qhShift;
    };
    const QuadInfo quads[4] = {
        {0, 0, 0}, {1, 0, 2}, {0, 1, 4}, {1, 1, 6}};

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-q6_K output not pointer");
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
    std::string vsllCallee = ("__riscv_vsll_vx_u8" + l8).str();
    std::string vorCallee = ("__riscv_vor_vv_u8" + l8).str();
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
    auto u8Or = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, u8mf2Type, vorCallee,
                            mlir::ValueRange{a, b, vl8}, opName, role);
    };
    auto qhQuadBits = [&](mlir::Value qh, int shift) -> mlir::Value {
      mlir::Value sel =
          (shift == 0) ? qh : u8Imm(vsrlCallee, qh, std::to_string(shift));
      return u8Imm(vsllCallee, u8Imm(vandCallee, sel, "0x03"), "4");
    };
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    std::string vsubI8Callee = ("__riscv_vsub_vx_i8" + l8).str();
    auto subBias = [&](mlir::Value i8vec) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vsubI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "32")
                    .getResult();
            return {i8vec, immV, vl8};
          });
    };
    auto assembleWeight = [&](mlir::Value ql, mlir::Value qh, int highNib,
                              int qhShift) -> mlir::Value {
      mlir::Value nib = highNib ? u8Imm(vsrlCallee, ql, "4")
                                : u8Imm(vandCallee, ql, "0x0F");
      mlir::Value raw = u8Or(nib, qhQuadBits(qh, qhShift));
      return subBias(reinterpretToI8(raw));
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
          // vs the strip-INTERLEAVED ~64 that spilled (board objdump: 2068 vs1r/vl1r
          // whole-reg spills + 2210 csrr vlenb spill-offset). BYTE-EXACT INVARIANT:
          // every out[c][h] accumulates over the SAME blocks in the SAME (l,j,sh,k,p,q)
          // order; hoisting h from innermost to outermost only reorders the interleave
          // of INDEPENDENT (disjoint weight-byte / disjoint output) strips, never a
          // single output's fold order. numHalves is the VLEN-derived strip count (2
          // @VLEN128 where the 32-vreg budget forces tiling, 1 @VLEN256 where the tile
          // degenerates to a single pass == byte-identical to the old strip-inner emit).
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
            for (int64_t j = 0; j < nSuperHalves; ++j) {
              for (int64_t sh = 0; sh < 2; ++sh) {
                // SHARED signed scale strips per (quad, sub-half): sub-block index
                // j*8 + q*2 + sh. Reused across columns AND k=0,1.
                step("signed_scale_unpack");
                llvm::SmallVector<mlir::Value> scaleVal;
                for (int q = 0; q < 4; ++q) {
                  int64_t sIdx = j * 8 + q * 2 + sh;
                  mlir::Value s8 = loadI8Strip(
                      bl, weightScalesOffset + sIdx * 16 + h * half);
                  scaleVal.push_back(liftScaleToI16(s8));
                }

                for (int64_t k = 0; k < 2; ++k) {
                  // Per-column i16 partials acc[c][q] over the 8 chunk positions (THIS
                  // strip; the h dimension is now the enclosing strip-outer loop).
                  llvm::SmallVector<llvm::SmallVector<mlir::Value>> acc(
                      activationInterleave);
                  if (rolledMainTerm) {
                    // ---- ROLLED whole-K-nest main term ([GAP-EMIT-VSETVL-TAX] /
                    // [K-10] structural GEMM plan): the dominant per-8-position inner
                    // p-loop is materialized as ONE runtime emitc.for so gcc hoists the
                    // e8 vsetvli out of the hot body (collapse the vsetvli storm). The
                    // 4-column x per-strip x per-quadrant i16 partials accVar[c][h][q]
                    // (32 @ mf2/VLEN128) are carried as RESIDENT SSA-register
                    // VariableOps (seeded ABOVE the loop, load-accumulate-store INSIDE
                    // it). The runtime position p only shifts the base pointers: bl+p*16
                    // for the 16-way-interleaved weight strips, al+p*4 for the
                    // 4-column-interleaved q8_Kx4 activation; the compile-time remainder
                    // rides the strip offset. The 6-bit weight decode stays INSIDE the
                    // p-loop but OUTSIDE the column loop -> each weight is decoded ONCE
                    // and shared across all 4 activation columns (NO re-decode, NO tile
                    // narrowing -- the whole-K-nest, not a narrow output tile).
                    // BYTE-EXACT to the unrolled emit by construction: the vwmacc16
                    // accumulation order (p ascending, then column c, then quadrant q,
                    // then strip h) is IDENTICAL -- only the loop is materialized.
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
                    auto pLoop = rewriter.create<emitc::ForOp>(
                        loc, sizeLit(0), sizeLit(8), sizeLit(1),
                        /*bodyBuilder=*/nullptr);
                    {
                      mlir::OpBuilder::InsertionGuard ig(rewriter);
                      rewriter.setInsertionPointToStart(pLoop.getBody());
                      mlir::Value pv = pLoop.getInductionVar();
                      int64_t llBase = sh * 16 + k * 8; // ll = llBase + p
                      mlir::Value pv16 = rewriter.create<emitc::MulOp>(
                          loc, sizeType, pv, sizeLit(16));
                      mlir::Value blP = rewriter.create<emitc::AddOp>(
                          loc, weightPtrType, bl, pv16);
                      mlir::Value pv4 = rewriter.create<emitc::MulOp>(
                          loc, sizeType, pv, sizeLit(4));
                      mlir::Value alP = rewriter.create<emitc::AddOp>(
                          loc, activationPtrType, al, pv4);
                      // THIS strip's weight decode per p: 4 quadrant signed weights,
                      // reused across the interleaved activation columns.
                      llvm::SmallVector<mlir::Value> wq;
                      step("weight_ql_qh_addr");
                      mlir::Value qlA = loadU8Strip(
                          blP, weightQlOffset + (j * 64 + llBase) * 16 + h * half);
                      mlir::Value qlB = loadU8Strip(
                          blP, weightQlOffset + (j * 64 + 32 + llBase) * 16 +
                                   h * half);
                      mlir::Value qh = loadU8Strip(
                          blP, weightQhOffset + (j * 32 + llBase) * 16 + h * half);
                      for (int q = 0; q < 4; ++q) {
                        mlir::Value qlSel =
                            (quads[q].qlStream == 0) ? qlA : qlB;
                        wq.push_back(assembleWeight(
                            qlSel, qh, quads[q].highNib, quads[q].qhShift));
                      }
                      for (int64_t c = cLo; c < cHi; ++c) {
                        step("act_quant_addr");
                        for (int q = 0; q < 4; ++q) {
                          int64_t gpBase = j * 128 + q * 32 + llBase;
                          mlir::Value aqc = i8Read(
                              alP, activationQuantOffset + gpBase * 4 + c);
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
                    // ---- UNROLLED main term (default, register-resident full
                    // unroll; the frozen shipped form). THIS strip only (the h
                    // dimension is the enclosing strip-outer loop). ----
                    for (int64_t c = cLo; c < cHi; ++c) {
                      for (int q = 0; q < 4; ++q)
                        acc[c].push_back(seedI16());
                    }

                    for (int64_t p = 0; p < 8; ++p) {
                      int64_t ll = sh * 16 + k * 8 + p; // 0..31 within quadrant
                      // THIS strip's weight decode per p: 4 quadrant signed weights,
                      // reused across the interleaved activation columns.
                      llvm::SmallVector<mlir::Value> wq;
                      step("weight_ql_qh_addr");
                      mlir::Value qlA = loadU8Strip(
                          bl, weightQlOffset + (j * 64 + ll) * 16 + h * half);
                      mlir::Value qlB = loadU8Strip(
                          bl,
                          weightQlOffset + (j * 64 + 32 + ll) * 16 + h * half);
                      mlir::Value qh = loadU8Strip(
                          bl, weightQhOffset + (j * 32 + ll) * 16 + h * half);
                      for (int q = 0; q < 4; ++q) {
                        mlir::Value qlSel =
                            (quads[q].qlStream == 0) ? qlA : qlB;
                        wq.push_back(assembleWeight(
                            qlSel, qh, quads[q].highNib, quads[q].qhShift));
                      }
                      for (int64_t c = cLo; c < cHi; ++c) {
                        step("act_quant_addr");
                        // Interleaved q8_Kx4 quant byte for column c, global element
                        // position gp = j*128 + q*32 + ll: offset + gp*4 + c.
                        for (int q = 0; q < 4; ++q) {
                          int64_t gp = j * 128 + q * 32 + ll;
                          mlir::Value aqc = i8Read(
                              al, activationQuantOffset + gp * 4 + c);
                          acc[c][q] = vwmacc16(acc[c][q], aqc, wq[q]);
                        }
                      }
                    }
                  }
                  // sumi_c += sum over 4 quadrants of signed_scale * acc_c (this strip).
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
    // result-less (the repacked lane-wise q6_K GEMM sinks through the output
    // pointer), so unlike the retired direct emitter this body leaf seeds NO dead
    // i32m1 result token.
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
