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

// Typed q4 scale/min K-quant repack artifact consumers.

// q4_K GEVM emitter (stage-1b, COMPLETE — this is the REAL K-quant decode, NOT q4_1-style).
// Emits the full super-block decode: the per-sub-block 6-bit scale/min are unpacked LANE-WISE
// (vand 0x0F / vsrl / vsll bit-dance), the MIN correction folds the activation int16 bsums
// weighted by the 6-bit mins, and each 32-elem sub-block dot is split into 2x16 i16 chunks
// (i16 overflow guard) promoted to i32 weighted by the 6-bit scale — byte-for-byte the lane-wise
// form of ggml arch/riscv/repack.cpp:260-389 (ggml_gemv_q4_K_16x1_q8_K). NUMERIC STATUS:
// oracle-verified vs an INDEPENDENT scalar dequant-matmul reference on the RVV1.0 VLEN128 board
// (bounded-norm PASS, WORST_NORM ~7e-07 over 8 shapes; controls NOMIN 4e5x / PERM 2e6x). NOT
// byte-exact vs ggml: the MIN term is folded once across super-halves (an IEEE-legal reassociation
// of ggml's per-super-half subtract), so the difference is fp round-off only, not a decode error.
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemvBodyQ4K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t activationQuantOffset,
    int64_t weightDminOffset, int64_t weightScalesOffset,
    int64_t activationBsumsOffset, int64_t nSubblocks, int64_t weightInterleave,
    int64_t half, bool rolledMainTerm) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one
    // notch i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
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
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    // The decode runs on the UNSIGNED weight nibble (q4_K stores RAW 4-bit
    // quants with NO offset-binary bias; the bias lives in the per-sub-block
    // 6-bit MIN); the activations stay i8.
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
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int16_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked q4_K (K-quant super-block) GEVM block-format structural
    // facts (I4 mirror, pinned by the verifier): QK_K=256, block_q4_Kx16 weight
    // stride 2304 (16 fp16 d + 16 fp16 dmin + 192 6-bit scales/mins + 2048
    // nibble bytes), block_q8_K activation stride 292 (fp32 d + 256 int8 quants
    // + 16 int16 bsums), the weight nibble bytes at +256, the per-column dmin
    // strip at +32, the custom 6-bit scales/mins region at +64, the activation
    // int8 quants at +4, the activation bsums at +260, 16 weight columns per
    // group, 8 sub-blocks of 32, and the VLEN-derived e8 half width. These are
    // PARAMETERS now (the loop body op's pinned attrs, read by the K-quant branch
    // of emitTypedRepackGemvLoopBody and passed in): qk, weightStride,
    // activationStride, weightQuantOffset, activationQuantOffset, weightDminOffset,
    // weightScalesOffset, activationBsumsOffset, nSubblocks, weightInterleave, half.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t subBlockSize = qk / nSubblocks;          // 32 elems / sub-block
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 4 sub-blocks / super-half
    (void)subBlockSize;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-q4_K output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width.
    mlir::Value vl8 = sizeLit(half);

    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");

    // size_t nb = n / QK_K;
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    // size_t nc_groups = nc / 16;
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load helpers (all UNSIGNED u8 contiguous strip loads) ----
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // vand_vx_u8 / vsrl_vx_u8 / vsll_vx_u8 -- the 6-bit scale/min unpack and the
    // 4-bit nibble decode bit-dance (UNSIGNED, value-identity reinterpret).
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
    // Reinterpret an unsigned 6-bit strip to a SIGNED i16 lane: the per-sub-block
    // 6-bit scale/min are 0..63 (value-identity), widened to i16 via vwmacc /
    // i32 widen below.
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    // A scalar i8 read of the activation quant byte a.qs[k] (int8).
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    // A scalar i16 read of the activation bsum a.bsums[k] (int16).
    llvm::StringRef i16ReadCallee = "*(const int16_t *)";
    auto i16Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i16ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_bsum_scalar"));
    };
    // vwmacc_vx i8->i16: acc += scalar * vec (the 4-bit nibble dot chunk).
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    // vwmacc_vv i16->i32: acc += scale_strip(i16) * sumi_s(i16). The per-sub-block
    // scale multiplies the per-sub-block i16 partial into the i32 accumulator.
    std::string vwmaccVV32Callee = ("__riscv_vwmacc_vv_i32" + l32).str();
    auto vwmaccVV32 = [&](mlir::Value acc, mlir::Value scaleStrip,
                          mlir::Value sumiS) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVV32Callee,
                            mlir::ValueRange{acc, scaleStrip, sumiS, vl8}, opName,
                            role);
    };
    // vwmacc_vx i16->i32: bsums_acc += bsum_pair_scalar * min_strip(i16). The
    // per-sub-block 6-bit min, weighted by the paired activation bsum.
    std::string vwmaccVX32Callee = ("__riscv_vwmacc_vx_i32" + l32).str();
    auto vwmaccVX32 = [&](mlir::Value acc, mlir::Value scalar,
                          mlir::Value minStrip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVX32Callee,
                            mlir::ValueRange{acc, scalar, minStrip, vl8}, opName,
                            role);
    };
    // vzext_vf2 u8 strip -> u16 strip, then reinterpret to i16: the 6-bit
    // scale/min (0..63) lifted to the i16 lane the vwmacc consumes.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string reinterpretU16I16Callee =
        ("__riscv_vreinterpret_v_u16" + l16 + "_i16" + l16).str();
    auto liftToI16 = [&](mlir::Value u8strip) -> mlir::Value {
      mlir::Value u16 =
          emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                         mlir::ValueRange{u8strip, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, i16m1Type, reinterpretU16I16Callee,
                            mlir::ValueRange{u16}, opName, role);
    };

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

      // const uint8_t *b = vx + x*nb*2304;  (the q4_Kx16 column group base).
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
        mlir::Value l = blockLoop.getInductionVar();

        // const uint8_t *bl = b + l*2304;   const uint8_t *al = a + l*292;
        step("weight_block_base");
        mlir::Value blOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(weightStride));
        mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                       bGroup, blOff);
        step("act_block_base");
        mlir::Value alOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(activationStride));
        mlir::Value al = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, aBase, alOff);

        // -- the activation super-block delta d_y = *(const float *)&al.d. q8_K
        // stores a FLOAT (4 bytes) at offset 0 (NOT the q8_1 _Float16). It scales
        // BOTH the main d term and the dmin MIN term.
        llvm::StringRef floatReadCallee = "*(const float *)";
        mlir::Type floatPtrConstType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
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

        // -- per-strip dmin_d_h = vfwcvt(vle16(&bl.dmin[h*half])) * d_y: the
        // super-block dmin (fp16 per column lane) widened to f32 and scaled by
        // the activation float delta. (MIN term scale.)
        std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
        auto loadF16Strip = [&](int64_t baseByteOff,
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
        llvm::SmallVector<mlir::Value> dminsD;
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dminStrip = loadF16Strip(weightDminOffset, h * half);
          dminsD.push_back(fmulScalar(widenF16(dminStrip), aD));
        }

        // ===== Per-block i32 accumulators per strip (scale main term). =====
        std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
        auto seedI32 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, i32m2Type, mvI32Callee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {zero, vl8};
              });
        };
        llvm::SmallVector<mlir::Value> sumiVar, bsumsVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
          auto bv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, bv, seedI32());
          bsumsVar.push_back(bv);
        }

        // i16 partial seed for the inner nibble dot.
        std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
        auto seedI16 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, i16m1Type, mvI16Callee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {zero, vl8};
              });
        };

        // ===== Super-half loop: for (j = 0; j < QK_K/128; ++j) =====
        // Each super-half is 4 sub-blocks. We UNPACK the 4 per-sub-block 6-bit
        // scale strips + 4 min strips LANE-WISE (vand 0x0F / vsrl / vsll bit
        // dance), then run the integer nibble dot, then the MIN bsums fold.
        for (int64_t j = 0; j < nSuperHalves; ++j) {
          step("scale_min_unpack_superhalf");
          // Per strip h, per local sub-block sb (0..3): the LOW nibble byte is at
          // scales[j*64 + sb*16 + h*half], the HIGH 2-bit byte at scales[128 +
          // sb*16 + h*half]. ggml's lane-wise q4_K unpack (arch/riscv/repack.cpp
          // 299-315): scale = (hi-bits) | (lo & 0x0F); min = (hi-bits) | (lo>>4).
          // scales_lo = lo & 0x0F; mins_lo = lo >> 4.
          // j==0: scales_hi = (hi & 0x03) << 4; mins_hi = (hi & 0x0C) << 2.
          // j==1: scales_hi =  hi & 0x30;       mins_hi = (hi & 0xC0) >> 2.
          // scaleVal[h][sb], minVal[h][sb] are i16 strips ready for vwmacc.
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> scaleVal(numHalves);
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> minVal(numHalves);
          for (int64_t h = 0; h < numHalves; ++h) {
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t loByte =
                  weightScalesOffset + j * 64 + sb * 16 + h * half;
              int64_t hiByte = weightScalesOffset + 128 + sb * 16 + h * half;
              mlir::Value lo = loadU8Strip(bl, sizeLit(loByte));
              mlir::Value hi = loadU8Strip(bl, sizeLit(hiByte));
              mlir::Value scalesLo = u8Imm(vandCallee, lo, "0x0F");
              mlir::Value minsLo = u8Imm(vsrlCallee, lo, "4");
              mlir::Value scalesHi, minsHi;
              if (j == 0) {
                scalesHi =
                    u8Imm(vsllCallee, u8Imm(vandCallee, hi, "0x03"), "4");
                minsHi = u8Imm(vsllCallee, u8Imm(vandCallee, hi, "0x0C"), "2");
              } else {
                scalesHi = u8Imm(vandCallee, hi, "0x30");
                minsHi = u8Imm(vsrlCallee, u8Imm(vandCallee, hi, "0xC0"), "2");
              }
              mlir::Value scU8 = u8Or(scalesHi, scalesLo);
              mlir::Value mnU8 = u8Or(minsHi, minsLo);
              scaleVal[h].push_back(liftToI16(scU8));
              minVal[h].push_back(liftToI16(mnU8));
            }
          }

          // ----- MIN term: bsums_acc += bsum_pair_sb * min_sb (i32 widen). The
          // paired activation bsums a.bsums[2*sub]+a.bsums[2*sub+1] (int16) scale
          // the per-sub-block 6-bit min strip. Global sub index = j*4 + sb. -----
          step("min_bsums_fold");
          for (int64_t sb = 0; sb < subPerSuper; ++sb) {
            int64_t gsub = j * subPerSuper + sb;
            mlir::Value bs0 =
                i16Read(al, activationBsumsOffset + (gsub * 2) * 2);
            mlir::Value bs1 =
                i16Read(al, activationBsumsOffset + (gsub * 2 + 1) * 2);
            mlir::Value bsPair =
                rewriter.create<emitc::AddOp>(loc, i32Type, bs0, bs1);
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value curB =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[h])
                      .getResult();
              rewriter.create<emitc::AssignOp>(
                  loc, bsumsVar[h], vwmaccVX32(curB, bsPair, minVal[h][sb]));
            }
          }

          // ----- MAIN term: per local sub-block sb, the 32-element integer dot
          // split into 2x16 i16 chunks (i16 overflow guard: 32*127*15 > 32767),
          // promoted to i32 weighted by the 6-bit scale. Mirrors ggml's k-loop.
          // Weight nibble byte: qs[256 + j*1024 + sb*256(half within super) ...].
          // ggml routes one byte's LOW nibble to even sub-block, HIGH to odd; we
          // process the pair (sb even, sb+1) together reading each byte ONCE. -----
          for (int64_t pair = 0; pair < subPerSuper / 2; ++pair) {
            int64_t sbLo = pair * 2;       // even local sub-block (low nibble)
            int64_t sbHi = pair * 2 + 1;   // odd  local sub-block (high nibble)
            // byte base qs[256 + j*1024 + pair*512 + i*16]; activation low
            // a.qs[gj*128 + sbLo*32 + i], high a.qs[gj*128 + sbHi*32 + i].
            int64_t qsPairBase = weightQuantOffset + j * 1024 + pair * 512;
            int64_t aLoBase = activationQuantOffset + j * 128 + sbLo * 32;
            int64_t aHiBase = activationQuantOffset + j * 128 + sbHi * 32;
            // i16 OVERFLOW GUARD: a 32-element sub-block dot would overflow i16
            // (32*127*15 > 32767), so the 32 positions are split into 2x16-element
            // k-chunks; each chunk's i16 partial is promoted to i32 (vwmacc_vv,
            // scale-weighted) before the next chunk. Mirrors ggml's k-loop.
            for (int64_t k = 0; k < 2; ++k) {
              llvm::SmallVector<mlir::Value> sLo(numHalves), sHi(numHalves);
              if (rolledMainTerm) {
                // ---- ROLLED whole-K-nest main term ([GAP-EMIT-KQUANT-GEVM-TILE-
                // ROUNDTRIP] maturity lever, the q5_K sibling MINUS the qh inject): the
                // dominant per-16-element inner ii-loop is emitted as ONE runtime
                // emitc.for; the per-strip i16 partials sLoVar/sHiVar are carried as
                // RESIDENT SSA-register VariableOps (seeded ABOVE the loop, load-
                // accumulate-store INSIDE it -- the SAME resident-across-emitc.for pattern
                // the sumf/sumi/bsums accumulators use). The per-element nibble decode is
                // computed and CONSUMED within ONE iteration -- NEVER materialized to a
                // stack scratch tile. BYTE-EXACT to the unrolled emit by construction: the
                // vwmacc16 integer accumulation order (ii ascending, then h) is IDENTICAL --
                // only the loop is materialized instead of unrolled.
                llvm::SmallVector<mlir::Value> sLoVar(numHalves), sHiVar(numHalves);
                for (int64_t h = 0; h < numHalves; ++h) {
                  auto lv = rewriter.create<emitc::VariableOp>(
                      loc, emitc::LValueType::get(i16m1Type),
                      emitc::OpaqueAttr::get(ctx, ""));
                  rewriter.create<emitc::AssignOp>(loc, lv, seedI16());
                  sLoVar[h] = lv;
                  auto hv = rewriter.create<emitc::VariableOp>(
                      loc, emitc::LValueType::get(i16m1Type),
                      emitc::OpaqueAttr::get(ctx, ""));
                  rewriter.create<emitc::AssignOp>(loc, hv, seedI16());
                  sHiVar[h] = hv;
                }
                auto iiLoop = rewriter.create<emitc::ForOp>(
                    loc, sizeLit(0), sizeLit(16), sizeLit(1),
                    /*bodyBuilder=*/nullptr);
                {
                  mlir::OpBuilder::InsertionGuard ig(rewriter);
                  rewriter.setInsertionPointToStart(iiLoop.getBody());
                  mlir::Value iiv = iiLoop.getInductionVar();
                  // i = k*16 + ii; the runtime weight byte offset = C + ii*16, the runtime
                  // activation byte offset = C + ii (C = the k*16-shifted base).
                  mlir::Value iiv16 = rewriter.create<emitc::MulOp>(
                      loc, sizeType, iiv, sizeLit(16));
                  for (int64_t h = 0; h < numHalves; ++h) {
                    step("weight_nibble_addr");
                    mlir::Value wOff = rewriter.create<emitc::AddOp>(
                        loc, sizeType,
                        sizeLit(qsPairBase + k * 256 + h * half), iiv16);
                    mlir::Value packed = loadU8Strip(bl, wOff);
                    mlir::Value nLo =
                        reinterpretToI8(u8Imm(vandCallee, packed, "0x0F"));
                    mlir::Value nHi =
                        reinterpretToI8(u8Imm(vsrlCallee, packed, "4"));
                    step("act_quant_addr");
                    mlir::Value aLoOff = rewriter.create<emitc::AddOp>(
                        loc, sizeType, sizeLit(aLoBase + k * 16), iiv);
                    mlir::Value aHiOff = rewriter.create<emitc::AddOp>(
                        loc, sizeType, sizeLit(aHiBase + k * 16), iiv);
                    mlir::Value aLo = i8Read(al, aLoOff);
                    mlir::Value aHi = i8Read(al, aHiOff);
                    mlir::Value curLo =
                        rewriter.create<emitc::LoadOp>(loc, i16m1Type, sLoVar[h])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(
                        loc, sLoVar[h], vwmacc16(curLo, aLo, nLo));
                    mlir::Value curHi =
                        rewriter.create<emitc::LoadOp>(loc, i16m1Type, sHiVar[h])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(
                        loc, sHiVar[h], vwmacc16(curHi, aHi, nHi));
                  }
                }
                for (int64_t h = 0; h < numHalves; ++h) {
                  sLo[h] =
                      rewriter.create<emitc::LoadOp>(loc, i16m1Type, sLoVar[h])
                          .getResult();
                  sHi[h] =
                      rewriter.create<emitc::LoadOp>(loc, i16m1Type, sHiVar[h])
                          .getResult();
                }
              } else {
                // ---- UNROLLED main term (default, register-resident full unroll) ----
                for (int64_t h = 0; h < numHalves; ++h) {
                  sLo[h] = seedI16();
                  sHi[h] = seedI16();
                }
                for (int64_t ii = 0; ii < 16; ++ii) {
                  int64_t i = k * 16 + ii;
                  for (int64_t h = 0; h < numHalves; ++h) {
                    step("weight_nibble_addr");
                    mlir::Value packed = loadU8Strip(
                        bl, sizeLit(qsPairBase + i * 16 + h * half));
                    mlir::Value nLo =
                        reinterpretToI8(u8Imm(vandCallee, packed, "0x0F"));
                    mlir::Value nHi =
                        reinterpretToI8(u8Imm(vsrlCallee, packed, "4"));
                    step("act_quant_addr");
                    mlir::Value aLo = i8Read(al, sizeLit(aLoBase + i));
                    mlir::Value aHi = i8Read(al, sizeLit(aHiBase + i));
                    sLo[h] = vwmacc16(sLo[h], aLo, nLo);
                    sHi[h] = vwmacc16(sHi[h], aHi, nHi);
                  }
                }
              }
              // sumi += scale_sbLo * sLo + scale_sbHi * sHi (i16->i32 vwmacc_vv).
              step("scale_subblock_fold");
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value cur0 =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                        .getResult();
                mlir::Value acc0 =
                    vwmaccVV32(cur0, scaleVal[h][sbLo], sLo[h]);
                rewriter.create<emitc::AssignOp>(
                    loc, sumiVar[h], vwmaccVV32(acc0, scaleVal[h][sbHi], sHi[h]));
              }
            }
          }
        }

        // ===== End-of-block fold per strip: sumf += d_x*d_y*sumi (main) then
        // sumf -= dmins_d*bsums (MIN). =====
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
          return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                mlir::ValueRange{v, vl8}, opName, role);
        };
        std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        std::string vfnmsacVVCallee = ("__riscv_vfnmsac_vv_f32" + l32).str();
        for (int64_t h = 0; h < numHalves; ++h) {
          // d_0 = vfwcvt(vle16(&bl.d[h*half])) * d_y; sumf += cvt(sumi) * d_0.
          mlir::Value dStrip = loadF16Strip(0, h * half);
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
          // sumf -= dmins_d * cvt(bsums)  (vfnmsac: acc -= a*b).
          mlir::Value bsumsV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[h])
                  .getResult();
          mlir::Value afterMin = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfnmsacVVCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {afterMain, dminsD[h], cvtI32F32(bsumsV), vl8};
              });
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], afterMin);
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
    // RESULT-LESS (no monolith token): the typed_repack_gemv_loop_body region is
    // result-less (the repacked lane-wise K-quant GEVM sinks through the output
    // pointer, not an SSA vector), so unlike the retired direct emitter this body
    // leaf seeds NO dead i32m1 result token.
    return mlir::success();
  }

// The INDEPENDENT q4_K colgroup-tiled GEVM Emission Plan BODY emitter ([K-10]
// structural-level Â· [PAT-2] P9). BYTE-EXACT to the sibling per-column GEVM plan
// (emitRepackKQuantGemvBodyQ4K) by construction: the per-block q4_K super-block
// decode + fold leaf (emitBlockFold below) is the SAME 8-sub-block 6-bit scale/min
// lane-wise unpack + split-32 main dot + bsums-min correction + dual d/dmin fp16
// fold. The ONLY difference is the ENVELOPE ([K-10] iteration-space topology): the
// weight-column-group loop is TILED by columnGroupTile (TG), the contraction-BLOCK
// loop is the shared MIDDLE loop (ONE q8_K activation block base + delta d_y per
// block, REUSED across the TG groups of the tile), the TG*numHalves per-strip f32
// accumulators are a REGISTER-RESIDENT bank live across the block stream, and the
// next block's weight strips are PREFETCHED one block ahead. A fixed output column's
// per-block reduction order is untouched (independent-column reordering) => byte-exact.
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemvColgroupTiledBodyQ4K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t activationQuantOffset,
    int64_t weightDminOffset, int64_t weightScalesOffset,
    int64_t activationBsumsOffset, int64_t nSubblocks, int64_t weightInterleave,
    int64_t half, int64_t columnGroupTile) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one
    // notch i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
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
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    // The decode runs on the UNSIGNED weight nibble (q4_K stores RAW 4-bit
    // quants with NO offset-binary bias; the bias lives in the per-sub-block
    // 6-bit MIN); the activations stay i8.
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
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int16_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked q4_K (K-quant super-block) GEVM block-format structural
    // facts (I4 mirror, pinned by the verifier): QK_K=256, block_q4_Kx16 weight
    // stride 2304 (16 fp16 d + 16 fp16 dmin + 192 6-bit scales/mins + 2048
    // nibble bytes), block_q8_K activation stride 292 (fp32 d + 256 int8 quants
    // + 16 int16 bsums), the weight nibble bytes at +256, the per-column dmin
    // strip at +32, the custom 6-bit scales/mins region at +64, the activation
    // int8 quants at +4, the activation bsums at +260, 16 weight columns per
    // group, 8 sub-blocks of 32, and the VLEN-derived e8 half width. These are
    // PARAMETERS now (the loop body op's pinned attrs, read by the K-quant branch
    // of emitTypedRepackGemvLoopBody and passed in): qk, weightStride,
    // activationStride, weightQuantOffset, activationQuantOffset, weightDminOffset,
    // weightScalesOffset, activationBsumsOffset, nSubblocks, weightInterleave, half.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t subBlockSize = qk / nSubblocks;          // 32 elems / sub-block
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 4 sub-blocks / super-half
    (void)subBlockSize;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-q4_K output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width.
    mlir::Value vl8 = sizeLit(half);

    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");

    // size_t nb = n / QK_K;
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    // size_t nc_groups = nc / 16;
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load helpers (all UNSIGNED u8 contiguous strip loads) ----
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // vand_vx_u8 / vsrl_vx_u8 / vsll_vx_u8 -- the 6-bit scale/min unpack and the
    // 4-bit nibble decode bit-dance (UNSIGNED, value-identity reinterpret).
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
    // Reinterpret an unsigned 6-bit strip to a SIGNED i16 lane: the per-sub-block
    // 6-bit scale/min are 0..63 (value-identity), widened to i16 via vwmacc /
    // i32 widen below.
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    // A scalar i8 read of the activation quant byte a.qs[k] (int8).
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    // A scalar i16 read of the activation bsum a.bsums[k] (int16).
    llvm::StringRef i16ReadCallee = "*(const int16_t *)";
    auto i16Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i16ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_bsum_scalar"));
    };
    // vwmacc_vx i8->i16: acc += scalar * vec (the 4-bit nibble dot chunk).
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    // vwmacc_vv i16->i32: acc += scale_strip(i16) * sumi_s(i16). The per-sub-block
    // scale multiplies the per-sub-block i16 partial into the i32 accumulator.
    std::string vwmaccVV32Callee = ("__riscv_vwmacc_vv_i32" + l32).str();
    auto vwmaccVV32 = [&](mlir::Value acc, mlir::Value scaleStrip,
                          mlir::Value sumiS) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVV32Callee,
                            mlir::ValueRange{acc, scaleStrip, sumiS, vl8}, opName,
                            role);
    };
    // vwmacc_vx i16->i32: bsums_acc += bsum_pair_scalar * min_strip(i16). The
    // per-sub-block 6-bit min, weighted by the paired activation bsum.
    std::string vwmaccVX32Callee = ("__riscv_vwmacc_vx_i32" + l32).str();
    auto vwmaccVX32 = [&](mlir::Value acc, mlir::Value scalar,
                          mlir::Value minStrip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVX32Callee,
                            mlir::ValueRange{acc, scalar, minStrip, vl8}, opName,
                            role);
    };
    // vzext_vf2 u8 strip -> u16 strip, then reinterpret to i16: the 6-bit
    // scale/min (0..63) lifted to the i16 lane the vwmacc consumes.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string reinterpretU16I16Callee =
        ("__riscv_vreinterpret_v_u16" + l16 + "_i16" + l16).str();
    auto liftToI16 = [&](mlir::Value u8strip) -> mlir::Value {
      mlir::Value u16 =
          emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                         mlir::ValueRange{u8strip, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, i16m1Type, reinterpretU16I16Callee,
                            mlir::ValueRange{u16}, opName, role);
    };

    // The DISTINGUISHING structural fact: TG weight-column-groups per tile.
    int64_t TG = columnGroupTile;
    step("colgroup_tile_count");
    mlir::Value nTiles =
        rewriter.create<emitc::DivOp>(loc, sizeType, ncGroups, sizeLit(TG));

    // Per-strip f32 accumulator zero seed (vfmv_v_f(0.0f, half)).
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

    // ===== Per-block FOLD leaf (BYTE-EXACT q4_K super-block decode): folds ONE
    // contraction super-block of ONE column-group (weight base bl) into the given
    // per-strip f32 accumulator bank sumfVar, using the SHARED q8_K activation block
    // base al + delta aD. IDENTICAL arithmetic to the sibling per-column GEVM plan;
    // only the caller's iteration ENVELOPE differs ([K-10] structural). =====
    auto emitBlockFold = [&](mlir::Value bl, mlir::Value al, mlir::Value aD,
                             llvm::SmallVector<mlir::Value> &sumfVar) {
        // -- per-strip dmin_d_h = vfwcvt(vle16(&bl.dmin[h*half])) * d_y: the
        // super-block dmin (fp16 per column lane) widened to f32 and scaled by
        // the activation float delta. (MIN term scale.)
        std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
        auto loadF16Strip = [&](int64_t baseByteOff,
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
        llvm::SmallVector<mlir::Value> dminsD;
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dminStrip = loadF16Strip(weightDminOffset, h * half);
          dminsD.push_back(fmulScalar(widenF16(dminStrip), aD));
        }

        // ===== Per-block i32 accumulators per strip (scale main term). =====
        std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
        auto seedI32 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, i32m2Type, mvI32Callee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {zero, vl8};
              });
        };
        llvm::SmallVector<mlir::Value> sumiVar, bsumsVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
          auto bv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, bv, seedI32());
          bsumsVar.push_back(bv);
        }

        // i16 partial seed for the inner nibble dot.
        std::string mvI16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
        auto seedI16 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, i16m1Type, mvI16Callee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {zero, vl8};
              });
        };

        // ===== Super-half loop: for (j = 0; j < QK_K/128; ++j) =====
        // Each super-half is 4 sub-blocks. We UNPACK the 4 per-sub-block 6-bit
        // scale strips + 4 min strips LANE-WISE (vand 0x0F / vsrl / vsll bit
        // dance), then run the integer nibble dot, then the MIN bsums fold.
        for (int64_t j = 0; j < nSuperHalves; ++j) {
          step("scale_min_unpack_superhalf");
          // Per strip h, per local sub-block sb (0..3): the LOW nibble byte is at
          // scales[j*64 + sb*16 + h*half], the HIGH 2-bit byte at scales[128 +
          // sb*16 + h*half]. ggml's lane-wise q4_K unpack (arch/riscv/repack.cpp
          // 299-315): scale = (hi-bits) | (lo & 0x0F); min = (hi-bits) | (lo>>4).
          // scales_lo = lo & 0x0F; mins_lo = lo >> 4.
          // j==0: scales_hi = (hi & 0x03) << 4; mins_hi = (hi & 0x0C) << 2.
          // j==1: scales_hi =  hi & 0x30;       mins_hi = (hi & 0xC0) >> 2.
          // scaleVal[h][sb], minVal[h][sb] are i16 strips ready for vwmacc.
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> scaleVal(numHalves);
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> minVal(numHalves);
          for (int64_t h = 0; h < numHalves; ++h) {
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t loByte =
                  weightScalesOffset + j * 64 + sb * 16 + h * half;
              int64_t hiByte = weightScalesOffset + 128 + sb * 16 + h * half;
              mlir::Value lo = loadU8Strip(bl, sizeLit(loByte));
              mlir::Value hi = loadU8Strip(bl, sizeLit(hiByte));
              mlir::Value scalesLo = u8Imm(vandCallee, lo, "0x0F");
              mlir::Value minsLo = u8Imm(vsrlCallee, lo, "4");
              mlir::Value scalesHi, minsHi;
              if (j == 0) {
                scalesHi =
                    u8Imm(vsllCallee, u8Imm(vandCallee, hi, "0x03"), "4");
                minsHi = u8Imm(vsllCallee, u8Imm(vandCallee, hi, "0x0C"), "2");
              } else {
                scalesHi = u8Imm(vandCallee, hi, "0x30");
                minsHi = u8Imm(vsrlCallee, u8Imm(vandCallee, hi, "0xC0"), "2");
              }
              mlir::Value scU8 = u8Or(scalesHi, scalesLo);
              mlir::Value mnU8 = u8Or(minsHi, minsLo);
              scaleVal[h].push_back(liftToI16(scU8));
              minVal[h].push_back(liftToI16(mnU8));
            }
          }

          // ----- MIN term: bsums_acc += bsum_pair_sb * min_sb (i32 widen). The
          // paired activation bsums a.bsums[2*sub]+a.bsums[2*sub+1] (int16) scale
          // the per-sub-block 6-bit min strip. Global sub index = j*4 + sb. -----
          step("min_bsums_fold");
          for (int64_t sb = 0; sb < subPerSuper; ++sb) {
            int64_t gsub = j * subPerSuper + sb;
            mlir::Value bs0 =
                i16Read(al, activationBsumsOffset + (gsub * 2) * 2);
            mlir::Value bs1 =
                i16Read(al, activationBsumsOffset + (gsub * 2 + 1) * 2);
            mlir::Value bsPair =
                rewriter.create<emitc::AddOp>(loc, i32Type, bs0, bs1);
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value curB =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[h])
                      .getResult();
              rewriter.create<emitc::AssignOp>(
                  loc, bsumsVar[h], vwmaccVX32(curB, bsPair, minVal[h][sb]));
            }
          }

          // ----- MAIN term: per local sub-block sb, the 32-element integer dot
          // split into 2x16 i16 chunks (i16 overflow guard: 32*127*15 > 32767),
          // promoted to i32 weighted by the 6-bit scale. Mirrors ggml's k-loop.
          // Weight nibble byte: qs[256 + j*1024 + sb*256(half within super) ...].
          // ggml routes one byte's LOW nibble to even sub-block, HIGH to odd; we
          // process the pair (sb even, sb+1) together reading each byte ONCE. -----
          for (int64_t pair = 0; pair < subPerSuper / 2; ++pair) {
            int64_t sbLo = pair * 2;       // even local sub-block (low nibble)
            int64_t sbHi = pair * 2 + 1;   // odd  local sub-block (high nibble)
            // byte base qs[256 + j*1024 + pair*512 + i*16]; activation low
            // a.qs[gj*128 + sbLo*32 + i], high a.qs[gj*128 + sbHi*32 + i].
            int64_t qsPairBase = weightQuantOffset + j * 1024 + pair * 512;
            int64_t aLoBase = activationQuantOffset + j * 128 + sbLo * 32;
            int64_t aHiBase = activationQuantOffset + j * 128 + sbHi * 32;
            // i16 OVERFLOW GUARD: a 32-element sub-block dot would overflow i16
            // (32*127*15 > 32767), so the 32 positions are split into 2x16-element
            // k-chunks; each chunk's i16 partial is promoted to i32 (vwmacc_vv,
            // scale-weighted) before the next chunk. Mirrors ggml's k-loop.
            for (int64_t k = 0; k < 2; ++k) {
              llvm::SmallVector<mlir::Value> sLo(numHalves), sHi(numHalves);
              for (int64_t h = 0; h < numHalves; ++h) {
                sLo[h] = seedI16();
                sHi[h] = seedI16();
              }
              for (int64_t ii = 0; ii < 16; ++ii) {
                int64_t i = k * 16 + ii;
                for (int64_t h = 0; h < numHalves; ++h) {
                  step("weight_nibble_addr");
                  mlir::Value packed = loadU8Strip(
                      bl, sizeLit(qsPairBase + i * 16 + h * half));
                  mlir::Value nLo =
                      reinterpretToI8(u8Imm(vandCallee, packed, "0x0F"));
                  mlir::Value nHi =
                      reinterpretToI8(u8Imm(vsrlCallee, packed, "4"));
                  step("act_quant_addr");
                  mlir::Value aLo = i8Read(al, sizeLit(aLoBase + i));
                  mlir::Value aHi = i8Read(al, sizeLit(aHiBase + i));
                  sLo[h] = vwmacc16(sLo[h], aLo, nLo);
                  sHi[h] = vwmacc16(sHi[h], aHi, nHi);
                }
              }
              // sumi += scale_sbLo * sLo + scale_sbHi * sHi (i16->i32 vwmacc_vv).
              step("scale_subblock_fold");
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value cur0 =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                        .getResult();
                mlir::Value acc0 =
                    vwmaccVV32(cur0, scaleVal[h][sbLo], sLo[h]);
                rewriter.create<emitc::AssignOp>(
                    loc, sumiVar[h], vwmaccVV32(acc0, scaleVal[h][sbHi], sHi[h]));
              }
            }
          }
        }

        // ===== End-of-block fold per strip: sumf += d_x*d_y*sumi (main) then
        // sumf -= dmins_d*bsums (MIN). =====
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
          return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                mlir::ValueRange{v, vl8}, opName, role);
        };
        std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        std::string vfnmsacVVCallee = ("__riscv_vfnmsac_vv_f32" + l32).str();
        for (int64_t h = 0; h < numHalves; ++h) {
          // d_0 = vfwcvt(vle16(&bl.d[h*half])) * d_y; sumf += cvt(sumi) * d_0.
          mlir::Value dStrip = loadF16Strip(0, h * half);
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
          // sumf -= dmins_d * cvt(bsums)  (vfnmsac: acc -= a*b).
          mlir::Value bsumsV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[h])
                  .getResult();
          mlir::Value afterMin = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfnmsacVVCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {afterMain, dminsD[h], cvtI32F32(bsumsV), vl8};
              });
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], afterMin);
        }
    };

    mlir::Value aBase = activationBase;
    mlir::LogicalResult status = mlir::success();

    // ===== Outer weight-column-group TILE loop: for (xt = 0; xt < nTiles; ++xt) =====
    auto tileLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nTiles,
                                                  sizeLit(1),
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard tg(rewriter);
      rewriter.setInsertionPointToStart(tileLoop.getBody());
      mlir::Value xt = tileLoop.getInductionVar();
      mlir::Value xtTG =
          rewriter.create<emitc::MulOp>(loc, sizeType, xt, sizeLit(TG));

      // TG weight-group bases + TG*numHalves REGISTER-RESIDENT f32 accumulators.
      llvm::SmallVector<mlir::Value> gIdx(TG), bGroup(TG);
      llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumfBank(TG);
      for (int64_t cg = 0; cg < TG; ++cg) {
        step("colgroup_tile_base");
        mlir::Value g = cg == 0 ? xtTG
                                : rewriter.create<emitc::AddOp>(
                                      loc, sizeType, xtTG, sizeLit(cg));
        gIdx[cg] = g;
        // const uint8_t *b = vx + (xt*TG + cg)*nb*2304 (the q4_Kx16 group base).
        step("weight_group_base");
        mlir::Value gBlocks = rewriter.create<emitc::MulOp>(loc, sizeType, g, nb);
        mlir::Value gOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, gBlocks, sizeLit(weightStride));
        bGroup[cg] = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                   weightBase, gOff);
        step("register_resident_bank_seed");
        for (int64_t h = 0; h < numHalves; ++h) {
          auto v = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(f32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, v, seedF32());
          sumfBank[cg].push_back(v);
        }
      }

      // ===== Shared contraction-BLOCK stream: for (l = 0; l < nb; ++l) =====
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard bg(rewriter);
        rewriter.setInsertionPointToStart(blockLoop.getBody());
        mlir::Value l = blockLoop.getInductionVar();

        // ONE q8_K activation block base al = a + l*292, SHARED across the TG groups.
        step("shared_act_block_base");
        mlir::Value alOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, l, sizeLit(activationStride));
        mlir::Value al = rewriter.create<emitc::AddOp>(loc, activationPtrType,
                                                       aBase, alOff);
        // aD = *(const float *)al (the SHARED activation super-block delta d_y).
        step("shared_act_scale_scalar");
        llvm::StringRef floatReadCallee = "*(const float *)";
        mlir::Type floatPtrConstType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
        mlir::Value aD = emitOpaqueCallBuilt(
            rewriter, loc, floatType, floatReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location ll) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, floatPtrConstType, al)
                      .getResult();
              return {aDCast};
            },
            llvm::StringRef("act_scale_scalar"));

        // PREFETCH cadence in the structure ([PAT-2] P9): hint the NEXT block's
        // weight strips for each column-group of the tile (the resident repack
        // layout). A pure hint -- never a value; __builtin_prefetch is fault-safe.
        mlir::Value lNext =
            rewriter.create<emitc::AddOp>(loc, sizeType, l, sizeLit(1));
        mlir::Value lNextOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, lNext, sizeLit(weightStride));
        for (int64_t cg = 0; cg < TG; ++cg) {
          step("weight_strip_prefetch");
          mlir::Value pf = rewriter.create<emitc::AddOp>(
              loc, weightPtrType, bGroup[cg], lNextOff);
          mlir::Value pfRw =
              rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0")
                  .getResult();
          mlir::Value pfLoc =
              rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                  .getResult();
          emitOpaqueCallVoid(rewriter, loc, "__builtin_prefetch",
                             mlir::ValueRange{pf, pfRw, pfLoc}, opName, role);
        }

        // Fold ONE block of EACH column-group in the tile into its resident bank
        // (the per-block leaf is byte-exact; the tile share the al/aD load above).
        for (int64_t cg = 0; cg < TG; ++cg) {
          step("weight_block_base");
          mlir::Value blOff = rewriter.create<emitc::MulOp>(
              loc, sizeType, l, sizeLit(weightStride));
          mlir::Value bl = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                         bGroup[cg], blOff);
          emitBlockFold(bl, al, aD, sumfBank[cg]);
        }
      }

      // ===== Per-tile store: TG groups x numHalves strips, vse32. =====
      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      auto storeHalf = [&](mlir::Value gColIdx, mlir::Value sumfStrip,
                           int64_t laneOff) {
        step("output_addr");
        mlir::Value x16 = rewriter.create<emitc::MulOp>(
            loc, sizeType, gColIdx, sizeLit(weightInterleave));
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
      for (int64_t cg = 0; cg < TG; ++cg)
        for (int64_t h = 0; h < numHalves; ++h)
          storeHalf(gIdx[cg], sumfBank[cg][h], h * half);
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    // RESULT-LESS: the colgroup-tiled K-quant GEVM plan sinks through the output
    // pointer (no SSA vector token), exactly like the sibling per-column GEVM plan.
    return mlir::success();
  }


// The INDEPENDENT q4_K colgroup-tiled GEVM Emission Plan dispatch ([K-10]
// structural-level · [PAT-2] P9). Gates on the in-region repack_gemv_kquant_core
// (decode_model "q4_K") anti-bypass brick (the SAME byte-exact contract as the
// sibling GEVM plan), reads the super-block decode facts + the DISTINGUISHING
// column_group_tile from the op's attrs, and re-emits the byte-exact q4_K decode
// through the column-group-tiled / block-streaming / register-resident-bank
// envelope.
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemmBodyQ4K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQuantOffset,
    int64_t activationQuantOffset, int64_t weightDminOffset,
    int64_t weightScalesOffset, int64_t activationBsumsOffset,
    int64_t nSubblocks, int64_t weightInterleave, int64_t activationInterleave,
    int64_t half, bool rolledMainTerm, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one notch
    // i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip. This is
    // the SAME parametric chain the q4_K GEVM sibling carries; the GEMM inherits
    // the already-LMUL-parametric 8-sub-block 6-bit unpack from it.
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
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    // The decode runs on the UNSIGNED weight nibble (q4_K stores RAW 4-bit quants
    // with NO offset-binary bias; the bias lives in the per-sub-block 6-bit MIN);
    // the activations stay i8.
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
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int16_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked q4_K (K-quant super-block) PREFILL GEMM block-format
    // structural facts (I4 mirror, pinned by the verifier): QK_K=256,
    // block_q4_Kx16 weight stride 2304 (16 fp16 d + 16 fp16 dmin + 192 6-bit
    // scales/mins + 2048 nibble bytes -- the SAME weight ABI as the GEVM),
    // block_q8_Kx4 activation stride 1168 (4 fp32 d + 1024 int8 quants [4 columns
    // interleaved] + 64 int16 bsums [16 per column * 4 columns]), the weight
    // nibble bytes at +256, the per-column dmin strip at +32, the custom 6-bit
    // scales/mins region at +64, the interleaved activation int8 quants at +16,
    // the interleaved activation bsums at +1040, 16 weight columns / 4 activation
    // columns per group, 8 sub-blocks of 32, and the VLEN-derived e8 half width.
    // These are PARAMETERS now (the loop body op's pinned attrs, read by the K-quant
    // branch of emitTypedRepackGemmLoopBody and passed in): qk, weightStride,
    // activationStride, weightQuantOffset, activationQuantOffset, weightDminOffset,
    // weightScalesOffset, activationBsumsOffset, nSubblocks, weightInterleave,
    // activationInterleave, half.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t subBlockSize = qk / nSubblocks;          // 32 elems / sub-block
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 4 sub-blocks / super-half
    (void)subBlockSize;
    // RVV1.0 (fractional chain) holds all 4 activation columns at once in one
    // pass; RVV0.7.1 (whole-LMUL chain) doubles every rung and folds ONE column
    // per pass to keep the per-pass live set bounded (the identical spill-avoid
    // rationale the q4_1/q4_0 GEMM document). The amortizing path on rvv (VLEN128
    // mf2) and K1 (VLEN256 mf2) is columnsPerPass==4: the 8-sub-block 6-bit unpack
    // + nibble decode happen ONCE per 16-weight group and are REUSED across the 4
    // activation columns -- the e2e-win-prefill amortization the single-column
    // GEVM cannot have.
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave;  // 1 @rvv07; 4 @rvv1.0

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-q4_K output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width.
    mlir::Value vl8 = sizeLit(half);

    // size_t nb = n / QK_K;
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    // size_t nr_groups = nr / 4;   size_t nc_groups = nc / 16;
    step("row_group_count");
    mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, rowCount, sizeLit(activationInterleave));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load helpers (UNSIGNED u8 contiguous strip loads) ----
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadU8Strip = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // vand_vx_u8 / vsrl_vx_u8 / vsll_vx_u8 -- the 6-bit scale/min unpack and the
    // 4-bit nibble decode bit-dance (UNSIGNED, value-identity reinterpret).
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
    // Reinterpret an unsigned 4-bit nibble strip to a SIGNED i8 lane (value-
    // identity for 0..15).
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    // A scalar i8 read of an interleaved activation quant byte a.qs[k] (int8).
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i8ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_quant_scalar"));
    };
    // A scalar i16 read of an interleaved activation bsum a.bsums[k] (int16).
    llvm::StringRef i16ReadCallee = "*(const int16_t *)";
    auto i16Read = [&](mlir::Value ab, int64_t byteOff) -> mlir::Value {
      mlir::Value full = ab;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, activationPtrType, ab,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, i16ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("act_bsum_scalar"));
    };
    // vwmacc_vx i8->i16: acc += scalar * vec (the 4-bit nibble dot chunk).
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    // vwmacc_vv i16->i32: acc += scale_strip(i16) * sumi_s(i16).
    std::string vwmaccVV32Callee = ("__riscv_vwmacc_vv_i32" + l32).str();
    auto vwmaccVV32 = [&](mlir::Value acc, mlir::Value scaleStrip,
                          mlir::Value sumiS) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVV32Callee,
                            mlir::ValueRange{acc, scaleStrip, sumiS, vl8}, opName,
                            role);
    };
    // vwmacc_vx i16->i32: bsums_acc += bsum_pair_scalar * min_strip(i16).
    std::string vwmaccVX32Callee = ("__riscv_vwmacc_vx_i32" + l32).str();
    auto vwmaccVX32 = [&](mlir::Value acc, mlir::Value scalar,
                          mlir::Value minStrip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVX32Callee,
                            mlir::ValueRange{acc, scalar, minStrip, vl8}, opName,
                            role);
    };
    // vzext_vf2 u8 strip -> u16 strip, then reinterpret to i16: the 6-bit
    // scale/min (0..63) lifted to the i16 lane the vwmacc consumes.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string reinterpretU16I16Callee =
        ("__riscv_vreinterpret_v_u16" + l16 + "_i16" + l16).str();
    auto liftToI16 = [&](mlir::Value u8strip) -> mlir::Value {
      mlir::Value u16 =
          emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                         mlir::ValueRange{u8strip, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, i16m1Type, reinterpretU16I16Callee,
                            mlir::ValueRange{u16}, opName, role);
    };
    // vfwcvt f16 strip -> f32 strip; vfmul_vf f32 strip by an fp32 scalar.
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
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                    .getResult();
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
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                    .getResult();
            return {zero, vl8};
          });
    };
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    auto cvtI32F32 = [&](mlir::Value v) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                            mlir::ValueRange{v, vl8}, opName, role);
    };
    std::string vfmaccVVCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    std::string vfnmsacVVCallee = ("__riscv_vfnmsac_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // ===== [M1c loop-interchange] SEL-1 loop-order schedule axis (PURE REALIZE) =====
    // The two group loops -- activation ROW-GROUP `y` (over nr/4) and weight
    // COLUMN-GROUP `x` (over nc/16) -- are INDEPENDENT: every out[y,x] is a private
    // K-accumulation, so either nesting order produces BYTE-IDENTICAL results and an
    // identical hot inner core. Which loop is OUTER is therefore a SCHEDULE axis keyed
    // on a LAYOUT/cache FACT (NOT a hardcoded constant). Under M1b this predicate was
    // INLINED here (`weightStride >= activationStride`); under M1c it is LIFTED to the
    // first-class SEL-1 loop-order selector (RVVRepackScheduleFormula,
    // stamped as loop_order). The caller passes the validated selected enum;
    // this body never reads attrs, reasons, measurements, or strides to select again.
    // Key rationale (unchanged): hold the DRAM-DOMINANT repacked
    // stream cache-resident across the hot inner sweep and restream the SMALLER one. For
    // the K-quant prefill GEMM the per-block weight panel (block_q4_Kx16 stride =
    // weightStride, 2304 B) is the larger stream vs the activation panel (block_q8_Kx4
    // stride = activationStride, 1168 B), so weightStride >= activationStride => the
    // col-group WEIGHT panel is made loop-OUTER (resident) and the row groups sweep
    // INSIDE it. Then the weight bytes of a col-group stream from DRAM ONCE instead of
    // once per row-group; the row-outer form re-reads that weight panel nr/4 (=32x on
    // the board) -- the H-B cold-stream root of the 0.764x e2e prefill gap (M0
    // attribution). The ROW-OUTER (activation-resident) nest is RETAINED as the other
    // arm for the layout where the activation panel is the larger stream.

    // a = vy + y*nb*activationStride  (the q8_Kx4 row-group base; a fn of y ONLY).
    auto emitAGroupBase = [&](mlir::Value y) -> mlir::Value {
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      return rewriter.create<emitc::AddOp>(loc, activationPtrType, activationBase,
                                           aGroupOff);
    };
    // b = vx + x*nb*weightStride  (the q4_Kx16 col-group base; a fn of x ONLY).
    // In the col-OUTER nest this is HOISTED above the row sweep (computed once per
    // col-group) so the weight panel base is loop-invariant across the row groups.
    auto emitBGroupBase = [&](mlir::Value x) -> mlir::Value {
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      return rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase,
                                           bGroupOff);
    };

    mlir::LogicalResult status = mlir::success();

    // The per-(y,x) OUTPUT TILE (cLo/h strip tiles -> block loop hot dot -> store).
    // BYTE-EXACT INVARIANT: this body is emitted IDENTICALLY for both loop orders --
    // only the two enclosing ForOp headers swap. Every out[y,x]'s K-accumulation and
    // end-of-block fold order is unchanged, so the S6 accumulator structure (peak
    // vreg <=32, hot-spill 0) and the objdump hot core (vwmacc multiset) are
    // invariant under the interchange.
    auto emitTile = [&](mlir::Value x, mlir::Value y, mlir::Value bGroup,
                        mlir::Value aGroup) {
        // Activation-column-PASS loop (compile-time, C++): the columns
        // [cLo, cLo+columnsPerPass) folded in this pass over the block loop.
        for (int64_t cLo = 0; cLo < activationInterleave;
             cLo += columnsPerPass) {
        int64_t cHi = cLo + columnsPerPass;

        // vfloat32m2_t sumf_{c,h} = vfmv_v_f(0.0f, half);  per column per strip,
        // carried across the contraction-block loop. The MIN correction is folded
        // straight into sumf via vfnmsac at end-of-block (no separate sum_minf).
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
        // ===== S1 h-strip OUTPUT TILE (mr=4 / hs=1) ======================
        // Process one DISJOINT weight strip h at a time, each in its OWN
        // contraction-block loop. This is a STRUCTURAL tile boundary, NOT a mere
        // emission reorder (the scheduler is free to re-interleave a single
        // fanned-out loop body, which is why the naive reorder is a spill NULL):
        // a SEPARATE emitc.for per strip scopes each block loop's live set, so the
        // register allocator only ever juggles ONE strip's accumulators + decode
        // temporaries at once (peak ~94 live vregs > 32 -> tile-local set, under
        // the spill cliff). The per-column f32 accumulator sumf_c is TILE-LOCAL: 4
        // accumulators, seeded ABOVE this strip's block loop and carried across it
        // as SSA-register VariableOps (NEVER rolled into the iter-arg-less
        // emitc.for -- the re-roll trap that memory-round-trips the accumulator).
        // Amortization is UNCHANGED: the h strips read DISJOINT weight bytes
        // (h*half offset), so nothing is shared across strips anyway, and the
        // weight decode stays amortized across the 4 activation columns
        // (columnsPerPass) WITHIN each strip. Capability-keyed schedule (an L2
        // property, not a value fact): numHalves == weight_interleave / half_lanes
        // is the VLEN-derived tile count -- 2 at VLEN128 (32-vreg budget forces
        // tiling), 1 at VLEN256 (the tile degenerates to a single pass,
        // byte-identical to the untiled emit).
        for (int64_t h = 0; h < numHalves; ++h) {
        // sumfVar[c] = vfmv_v_f(0.0f, half): THIS strip's per-column f32
        // accumulator, carried across the contraction-block loop. The MIN
        // correction is folded straight into sumf via vfnmsac at end-of-block (no
        // separate sum_minf).
        llvm::SmallVector<mlir::Value> sumfVar(activationInterleave);
        for (int64_t c = cLo; c < cHi; ++c) {
          auto v = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(f32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, v, seedF32());
          sumfVar[c] = v;
        }

        // ===== This strip's contraction-BLOCK loop: for (l = 0; l < nb; ++l) ==
        auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard bg(rewriter);
          rewriter.setInsertionPointToStart(blockLoop.getBody());
          mlir::Value l = blockLoop.getInductionVar();

          // const uint8_t *bl = b + l*2304;   const uint8_t *al = a + l*1168;
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

          // -- per-column activation super-block delta d_y_c = *(const float
          // *)&al.d[c]. block_q8_Kx4 stores 4 fp32 deltas at offsets 0,4,8,12
          // (one per interleaved column, NOT fp16). Each scales BOTH the main d
          // term and the dmin MIN term for ITS column.
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

          // ===== S6 decoded-weight + min-accumulator STACK PANELS (byte-exact) ===
          // The COLD decode band (this block's 6-bit scale/min i16 strips) and the
          // COLD per-column i32 MIN accumulator are staged into small PER-BLOCK
          // stack arrays, addressed via vle/vse opaque intrinsics (the taken
          // address defeats mem2reg, so they LIVE ON THE STACK, not in vregs). This
          // is the classic GEMM "pack the panel into a contiguous buffer" move
          // applied to the ALREADY-DECODED weights + the idle min accumulator: it
          // takes the scale strips (held live across the dot loop only for the
          // post-loop fold) and the bsums family (idle throughout the hot main
          // term, needed only at the end-of-block fold) OFF the hot live set, so
          // the allocator only juggles the HOT sumf/sumi accumulators + sLo/sHi
          // partials. Every staged value is INTEGER (exact store/reload) and the
          // end-of-block f32 fold is UNCHANGED => byte-exact. sumf/sumi stay SSA-
          // register accumulators (NEVER paneled, NEVER rolled into the iter-arg-
          // less emitc.for -- the re-roll trap). Declared PER-BLOCK (inside the
          // block loop) so each iteration fully writes-before-reads the panel (no
          // conservative prologue zero-init).
          mlir::Type i16ElemTy = emitc::OpaqueType::get(ctx, "int16_t");
          mlir::Type i32ElemTy = emitc::OpaqueType::get(ctx, "int32_t");
          mlir::Type i16MutPtrTy = emitc::PointerType::get(i16ElemTy);
          mlir::Type i32MutPtrTy = emitc::PointerType::get(i32ElemTy);
          mlir::Type scaleMinPanelTy =
              emitc::ArrayType::get({subPerSuper * half}, i16ElemTy);
          mlir::Type bsumsPanelTy =
              emitc::ArrayType::get({columnsPerPass * half}, i32ElemTy);
          auto mkPanel =
              [&](mlir::Type arrTy) -> mlir::TypedValue<emitc::ArrayType> {
            auto var = rewriter.create<emitc::VariableOp>(
                loc, arrTy, emitc::OpaqueAttr::get(ctx, ""));
            return llvm::cast<mlir::TypedValue<emitc::ArrayType>>(var.getResult());
          };
          // scale/min panels are per-super-half local (subPerSuper strips),
          // rewritten each super-half; bsums panel is per-block (columnsPerPass).
          mlir::TypedValue<emitc::ArrayType> scalePanel = mkPanel(scaleMinPanelTy);
          mlir::TypedValue<emitc::ArrayType> minPanel = mkPanel(scaleMinPanelTy);
          mlir::TypedValue<emitc::ArrayType> bsumsPanel = mkPanel(bsumsPanelTy);
          std::string vse16Callee = riscvIntrinsicName("vse", 16, l16, "i16");
          std::string vle16Callee = riscvIntrinsicName("vle", 16, l16, "i16");
          std::string vse32Callee = riscvIntrinsicName("vse", 32, l32, "i32");
          std::string vle32Callee = riscvIntrinsicName("vle", 32, l32, "i32");
          auto panelPtr = [&](mlir::TypedValue<emitc::ArrayType> arr,
                              mlir::Type ptrTy, int64_t idx) -> mlir::Value {
            mlir::Value iv = rewriter.create<emitc::LiteralOp>(
                loc, rewriter.getIndexType(), std::to_string(idx));
            mlir::Value elem =
                rewriter.create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{iv})
                    .getResult();
            return rewriter.create<emitc::ApplyOp>(loc, ptrTy, "&", elem)
                .getResult();
          };
          auto storeI16Panel = [&](mlir::TypedValue<emitc::ArrayType> arr,
                                   int64_t idx, mlir::Value vec) {
            emitOpaqueCallVoid(
                rewriter, loc, vse16Callee,
                mlir::ValueRange{panelPtr(arr, i16MutPtrTy, idx), vec, vl8}, opName,
                role);
          };
          auto loadI16Panel = [&](mlir::TypedValue<emitc::ArrayType> arr,
                                  int64_t idx) -> mlir::Value {
            return emitOpaqueCall(
                rewriter, loc, i16m1Type, vle16Callee,
                mlir::ValueRange{panelPtr(arr, i16MutPtrTy, idx), vl8}, opName,
                role);
          };
          auto storeBsumsPanel = [&](int64_t col, mlir::Value vec) {
            emitOpaqueCallVoid(
                rewriter, loc, vse32Callee,
                mlir::ValueRange{panelPtr(bsumsPanel, i32MutPtrTy, col * half), vec,
                                 vl8},
                opName, role);
          };
          auto loadBsumsPanel = [&](int64_t col) -> mlir::Value {
            return emitOpaqueCall(
                rewriter, loc, i32m2Type, vle32Callee,
                mlir::ValueRange{panelPtr(bsumsPanel, i32MutPtrTy, col * half),
                                 vl8},
                opName, role);
          };

          // -- S6 d/dmin ON-DEMAND: the SHARED (within the strip) per-column-lane
          // fp16 d/dmin strips of THIS h are per-block loop-invariant but are only
          // CONSUMED in the end-of-block f32 fold. Their widen-to-f32 is deferred
          // to that fold point (still loaded ONCE per block, byte-identical bytes)
          // so the two f32m2 strips do NOT sit live across the hot main-term dot
          // loop -- 4 vreg off the peak, byte-exact. (Moved down; see the fold.)

          // ===== Per-column i32 main + bsums accumulators for THIS strip. =====
          // sumiVar[c] (scale main term), bsumsVar[c] (min term).
          // sumiVar[c] stays a HOT SSA-register accumulator (scale main term). The
          // bsums (min term) accumulator family is STAGED to bsumsPanel -- idle
          // across the hot main dot and reloaded only at the end-of-block fold --
          // so it costs NO vreg on the peak. The panel is DEFINED by the first
          // sub-block's MIN accumulation (a register-seeded vwmacc, no separate
          // zero-store), so no broadcast-zero whole-register store is emitted.
          // Byte-exact: the i32 accumulation, single cvt, and fold order unchanged.
          llvm::SmallVector<mlir::Value> sumiVar(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            auto sv = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(i32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
            sumiVar[c] = sv;
          }

          // ===== Super-half loop: for (j = 0; j < QK_K/128; ++j) =====
          // Each super-half is 4 sub-blocks. UNPACK the 4 per-sub-block 6-bit
          // scale strips + 4 min strips LANE-WISE (SHARED across columns), then
          // run the per-column integer nibble dot, then the per-column MIN fold.
          for (int64_t j = 0; j < nSuperHalves; ++j) {
            step("scale_min_unpack_superhalf");
            // ggml's lane-wise q4_K unpack (arch/riscv/repack.cpp 299-315):
            //   scales_lo = lo & 0x0F; mins_lo = lo >> 4.
            //   j==0: scales_hi = (hi & 0x03) << 4; mins_hi = (hi & 0x0C) << 2.
            //   j==1: scales_hi =  hi & 0x30;       mins_hi = (hi & 0xC0) >> 2.
            // Unpacked ONCE for THIS strip; STAGED to scalePanel/minPanel (byte-
            // exact i16) and reloaded per column in the MIN/MAIN folds, so the
            // decode strips never sit live across the hot dot loop.
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t loByte = weightScalesOffset + j * 64 + sb * 16 + h * half;
              int64_t hiByte = weightScalesOffset + 128 + sb * 16 + h * half;
              mlir::Value lo = loadU8Strip(bl, sizeLit(loByte));
              mlir::Value hi = loadU8Strip(bl, sizeLit(hiByte));
              mlir::Value scalesLo = u8Imm(vandCallee, lo, "0x0F");
              mlir::Value minsLo = u8Imm(vsrlCallee, lo, "4");
              mlir::Value scalesHi, minsHi;
              if (j == 0) {
                scalesHi = u8Imm(vsllCallee, u8Imm(vandCallee, hi, "0x03"), "4");
                minsHi = u8Imm(vsllCallee, u8Imm(vandCallee, hi, "0x0C"), "2");
              } else {
                scalesHi = u8Imm(vandCallee, hi, "0x30");
                minsHi = u8Imm(vsrlCallee, u8Imm(vandCallee, hi, "0xC0"), "2");
              }
              mlir::Value scU8 = u8Or(scalesHi, scalesLo);
              mlir::Value mnU8 = u8Or(minsHi, minsLo);
              storeI16Panel(scalePanel, sb * half, liftToI16(scU8));
              storeI16Panel(minPanel, sb * half, liftToI16(mnU8));
            }

            // ----- MIN term per activation column m: bsums_acc[m] += bsum_pair *
            // min_sb. The interleaved q8_Kx4 bsums are group16-major/column-minor
            // (index = group16*4 + m); sub-block gsub spans groups 2*gsub and
            // 2*gsub+1, so the paired bsum for column m is
            //   a.bsums[gsub*8 + m] + a.bsums[gsub*8 + m + 4]
            // (the GEVM's gsub*2 / gsub*2+1 single-column pair, x4-interleaved).
            step("min_bsums_fold");
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t gsub = j * subPerSuper + sb;
              // min strip reloaded ONCE per sub-block from the panel, reused across
              // the 4 columns (same value ggml's minVal[sb] carried); the bsums
              // accumulator is read/updated straight in the panel (byte-exact i32).
              mlir::Value minStrip = loadI16Panel(minPanel, sb * half);
              for (int64_t c = cLo; c < cHi; ++c) {
                mlir::Value bs0 = i16Read(
                    al, activationBsumsOffset + (gsub * 8 + c) * 2);
                mlir::Value bs1 = i16Read(
                    al, activationBsumsOffset + (gsub * 8 + c + 4) * 2);
                mlir::Value bsPair =
                    rewriter.create<emitc::AddOp>(loc, i32Type, bs0, bs1);
                // First sub-block (gsub==0) DEFINES the panel from a register zero
                // seed (no zero-store); later sub-blocks load-accumulate-store.
                bool firstAcc = (j == 0 && sb == 0);
                mlir::Value cur =
                    firstAcc ? seedI32() : loadBsumsPanel(c - cLo);
                storeBsumsPanel(c - cLo, vwmaccVX32(cur, bsPair, minStrip));
              }
            }

            // ----- MAIN term: per local sub-block sb, the 32-element integer dot
            // split into 2x16 i16 chunks (i16 overflow guard: 32*127*15 > 32767),
            // promoted to i32 weighted by the 6-bit scale. The weight nibble is
            // DECODED ONCE per (i,h) and REUSED across the activation columns;
            // each column reads its own interleaved q8_Kx4 quant. ggml routes one
            // byte's LOW nibble to even sub-block, HIGH to odd (the pair shares a
            // byte). Activation low qs[16 + j*512 + sbLo*128 + i*4 + c], high
            // qs[16 + j*512 + sbHi*128 + i*4 + c] = low + 128.
            for (int64_t pair = 0; pair < subPerSuper / 2; ++pair) {
              int64_t sbLo = pair * 2;     // even local sub-block (low nibble)
              int64_t sbHi = pair * 2 + 1; // odd  local sub-block (high nibble)
              int64_t qsPairBase = weightQuantOffset + j * 1024 + pair * 512;
              int64_t aLoBase = activationQuantOffset + j * 512 + sbLo * 128;
              int64_t aHiBase = activationQuantOffset + j * 512 + sbHi * 128;
              for (int64_t k = 0; k < 2; ++k) {
                // Per-column i16 partials for this 16-element k-chunk (THIS h).
                llvm::SmallVector<mlir::Value> sLo(activationInterleave),
                    sHi(activationInterleave);
                if (rolledMainTerm) {
                  // ---- ROLLED whole-K-nest main term ([GAP-EMIT-VSETVL-TAX] /
                  // [K-10] structural GEMM plan): the dominant per-16-element inner
                  // ii-loop is materialized as ONE runtime emitc.for so gcc hoists the
                  // e8 vsetvli out of the hot body (collapse the vsetvli storm). The
                  // per-column i16 partials sLoVar/sHiVar are carried as RESIDENT
                  // SSA-register VariableOps (seeded ABOVE the loop, load-accumulate-
                  // store INSIDE it). The runtime position ii only shifts the byte
                  // offsets: the 16-way-interleaved weight nibble byte at C_w + ii*16,
                  // the 4-column-interleaved q8_Kx4 activation byte at C_a + ii*4; the
                  // compile-time remainder (k-chunk + h + column c) rides the base. The
                  // 4-bit nibble decode stays INSIDE the ii-loop but OUTSIDE the column
                  // loop -> each nibble is decoded ONCE and shared across all 4
                  // activation columns (NO re-decode, NO tile narrowing). ORTHOGONAL to
                  // colGroupOuter (the loop-order axis) and to the S6 stack panels
                  // (scale/min/bsums staging unchanged). BYTE-EXACT to the unrolled emit
                  // by construction: the vwmacc16 accumulation order (ii ascending, then
                  // column c) is IDENTICAL -- only the loop is materialized.
                  llvm::SmallVector<mlir::Value> sLoVar(activationInterleave),
                      sHiVar(activationInterleave);
                  for (int64_t c = cLo; c < cHi; ++c) {
                    auto lv = rewriter.create<emitc::VariableOp>(
                        loc, emitc::LValueType::get(i16m1Type),
                        emitc::OpaqueAttr::get(ctx, ""));
                    rewriter.create<emitc::AssignOp>(loc, lv, seedI16());
                    sLoVar[c] = lv;
                    auto hv = rewriter.create<emitc::VariableOp>(
                        loc, emitc::LValueType::get(i16m1Type),
                        emitc::OpaqueAttr::get(ctx, ""));
                    rewriter.create<emitc::AssignOp>(loc, hv, seedI16());
                    sHiVar[c] = hv;
                  }
                  auto iiLoop = rewriter.create<emitc::ForOp>(
                      loc, sizeLit(0), sizeLit(16), sizeLit(1),
                      /*bodyBuilder=*/nullptr);
                  {
                    mlir::OpBuilder::InsertionGuard ig(rewriter);
                    rewriter.setInsertionPointToStart(iiLoop.getBody());
                    mlir::Value iiv = iiLoop.getInductionVar();
                    // i = k*16 + ii; runtime weight byte = C_w + ii*16, runtime
                    // activation byte = C_a + ii*4 (C = the k*16-shifted base).
                    mlir::Value iiv16 = rewriter.create<emitc::MulOp>(
                        loc, sizeType, iiv, sizeLit(16));
                    mlir::Value iiv4 = rewriter.create<emitc::MulOp>(
                        loc, sizeType, iiv, sizeLit(4));
                    step("weight_nibble_addr");
                    mlir::Value wOff = rewriter.create<emitc::AddOp>(
                        loc, sizeType, sizeLit(qsPairBase + k * 256 + h * half),
                        iiv16);
                    mlir::Value packed = loadU8Strip(bl, wOff);
                    mlir::Value nLo =
                        reinterpretToI8(u8Imm(vandCallee, packed, "0x0F"));
                    mlir::Value nHi =
                        reinterpretToI8(u8Imm(vsrlCallee, packed, "4"));
                    for (int64_t c = cLo; c < cHi; ++c) {
                      step("act_quant_addr");
                      mlir::Value aLoOff = rewriter.create<emitc::AddOp>(
                          loc, sizeType, sizeLit(aLoBase + k * 64 + c), iiv4);
                      mlir::Value aHiOff = rewriter.create<emitc::AddOp>(
                          loc, sizeType, sizeLit(aHiBase + k * 64 + c), iiv4);
                      mlir::Value aLo = i8Read(al, aLoOff);
                      mlir::Value aHi = i8Read(al, aHiOff);
                      mlir::Value curLo =
                          rewriter.create<emitc::LoadOp>(loc, i16m1Type, sLoVar[c])
                              .getResult();
                      rewriter.create<emitc::AssignOp>(
                          loc, sLoVar[c], vwmacc16(curLo, aLo, nLo));
                      mlir::Value curHi =
                          rewriter.create<emitc::LoadOp>(loc, i16m1Type, sHiVar[c])
                              .getResult();
                      rewriter.create<emitc::AssignOp>(
                          loc, sHiVar[c], vwmacc16(curHi, aHi, nHi));
                    }
                  }
                  for (int64_t c = cLo; c < cHi; ++c) {
                    sLo[c] =
                        rewriter.create<emitc::LoadOp>(loc, i16m1Type, sLoVar[c])
                            .getResult();
                    sHi[c] =
                        rewriter.create<emitc::LoadOp>(loc, i16m1Type, sHiVar[c])
                            .getResult();
                  }
                } else {
                  // ---- UNROLLED main term (default when the [ROLL] resolver keeps the
                  // full static unroll; the register-resident form) ----
                  for (int64_t c = cLo; c < cHi; ++c) {
                    sLo[c] = seedI16();
                    sHi[c] = seedI16();
                  }
                  for (int64_t ii = 0; ii < 16; ++ii) {
                    int64_t i = k * 16 + ii;
                    // SHARED weight nibble decode per i (THIS h): reused over cols.
                    step("weight_nibble_addr");
                    mlir::Value packed =
                        loadU8Strip(bl, sizeLit(qsPairBase + i * 16 + h * half));
                    mlir::Value nLo =
                        reinterpretToI8(u8Imm(vandCallee, packed, "0x0F"));
                    mlir::Value nHi =
                        reinterpretToI8(u8Imm(vsrlCallee, packed, "4"));
                    for (int64_t c = cLo; c < cHi; ++c) {
                      step("act_quant_addr");
                      mlir::Value aLo =
                          i8Read(al, sizeLit(aLoBase + (i * 4 + c)));
                      mlir::Value aHi =
                          i8Read(al, sizeLit(aHiBase + (i * 4 + c)));
                      sLo[c] = vwmacc16(sLo[c], aLo, nLo);
                      sHi[c] = vwmacc16(sHi[c], aHi, nHi);
                    }
                  }
                }
                // sumi_c += scale_sbLo * sLo_c + scale_sbHi * sHi_c (i16->i32).
                // scale strips reloaded ONCE from the panel per (pair,k), reused
                // across the 4 columns (byte-exact i16, same scaleVal[sb]).
                step("scale_subblock_fold");
                mlir::Value scLo = loadI16Panel(scalePanel, sbLo * half);
                mlir::Value scHi = loadI16Panel(scalePanel, sbHi * half);
                for (int64_t c = cLo; c < cHi; ++c) {
                  mlir::Value cur0 =
                      rewriter
                          .create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                          .getResult();
                  mlir::Value acc0 = vwmaccVV32(cur0, scLo, sLo[c]);
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c], vwmaccVV32(acc0, scHi, sHi[c]));
                }
              }
            }
          }

          // ===== End-of-block fold per column per strip: sumf_c += d_x*d_y_c*sumi
          // (main) then sumf_c -= dmin_x*d_y_c*bsums_c (MIN). d_x/dmin_x are the
          // SHARED per-strip fp16 d/dmin widened to f32; multiplied by the
          // per-column fp32 d_y_c. =====
          // S6: the d/dmin widen is loaded HERE (on-demand at the fold), not at the
          // top of the block, so it does not occupy 4 vreg across the main dot loop.
          mlir::Value dminF32 =
              widenF16(loadF16Strip(bl, weightDminOffset, h * half));
          mlir::Value dF32 = widenF16(loadF16Strip(bl, 0, h * half));
          for (int64_t c = cLo; c < cHi; ++c) {
            // d_0_c = dF32 * d_y_c;  sumf_c += cvt(sumi_c) * d_0_c.
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
            // dmin_0_c = dminF32 * d_y_c;  sumf_c -= dmin_0_c * cvt(bsums_c).
            // bsums_c reloaded from its stack panel (byte-exact i32); the fold order
            // (main vfmacc THEN min vfnmsac) is identical to S1 -> bit-identical.
            mlir::Value dmin0 = fmulScalar(dminF32, aD[c]);
            mlir::Value bsumsV = loadBsumsPanel(c - cLo);
            mlir::Value afterMin = emitOpaqueCallBuilt(
                rewriter, loc, f32m2Type, vfnmsacVVCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  return {afterMain, dmin0, cvtI32F32(bsumsV), vl8};
                });
            rewriter.create<emitc::AssignOp>(loc, sumfVar[c], afterMin);
          }
        }

        // Per-column store for THIS strip: s + (y*4 + c)*bs + x*16 + h*half.
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
                             mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
        }
        } // end S1 h-strip output tile (hs=1)
        } // end activation-column-PASS loop (cLo)
    };  // end emitTile (identical body for both loop orders)

    if (colGroupOuter) {
      // ===== col-group WEIGHT panel OUTER / row groups sweep INSIDE (deployed) ==
      // for (x=0; x<nc/16; ++x) { bGroup=f(x);          // weight base HOISTED
      //   for (y=0; y<nr/4; ++y) { aGroup=f(y); emitTile(x,y,bGroup,aGroup); } }
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
      // ===== activation ROW panel OUTER / col groups sweep INSIDE (retained) ====
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
    // RESULT-LESS (no monolith token): the typed_repack_gemm_loop_body region is
    // result-less (the repacked lane-wise K-quant GEMM sinks through the output
    // pointer, not an SSA vector), so unlike the retired direct emitter this body
    // leaf seeds NO dead i32m1 result token.
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
