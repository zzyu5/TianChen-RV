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

// Typed low-bit q2 K-quant repack artifact consumers.

// q2_K x q8_K 16x1-REPACKED GEVM (decode) body leaf, from the FRONT DOOR. q2_K is the
// LOWEST-bit K-quant and the MIN-TERM regression: a FAITHFUL reuse of the oracle-
// verified q4_K super-block scaffold (LANE-WISE vwmacc block-as-lane strips, dual
// super-block d/dmin fold, the bsums-min correction -- emitRepackKQuantGemvBodyQ4K)
// with THREE q2_K deltas: (1) the weight is 2-BIT, FOUR lanes per byte -- the qs byte
// is loaded ONCE per column strip and peeled by vsrl {0,2,4,6} + vand 0x03 into four
// UNSIGNED [0,3] lanes (NO offset-binary bias; the bias lives entirely in the 4-bit
// MIN, exactly q4_K). (2) the per-sub-block scale/min is a SINGLE 4-bit-packed byte --
// vand 0x0F is the scale, vsrl 4 the min (NOT q4_K's 6-bit two-byte get_scale_min_k4
// bit-dance). (3) 16 sub-blocks of 16 -> each sub-block is EXACTLY ONE q8_K bsums
// group, so the MIN term reads a SINGLE bsum per sub-block (NOT q4_K's paired bs0+bs1)
// and each 16-element dot fits in ONE i16 partial (16*127*3 = 6096 < 32767, NO 2x16
// k-chunk split). The block-format facts are PARAMETERS (the loop body op's pinned
// attrs, read by the K-quant branch of emitTypedRepackGemvLoopBody and passed in).
// NUMERIC STATUS: oracle-verified byte-exact INTEGER isum + summs vs an INDEPENDENT
// scalar q2_K dequant-matmul reference. The q8_K activation ABI is byte-identical to
// q4_K/q5_K/q6_K. RESULT-LESS (no monolith token).
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemvBodyQ2K(
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

    // The integer-product core LMUL anchor (the *how*, never the *what*; the 16-way
    // interleaved repack reads the SAME bytes either way). "mf2" (default) is the
    // RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2 (f16 scale m1). "m1"
    // is the WHOLE-LMUL chain RVV0.7.1 requires: i8m1 -> i16m2 -> i32m4 -> f32m4.
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
    // The decode runs on the UNSIGNED weight 2-bit value (q2_K stores RAW 2-bit
    // quants with NO offset-binary bias; the bias lives in the per-sub-block 4-bit
    // MIN); the activations stay i8.
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

    // The 16x1 repacked q2_K (K-quant super-block) GEVM block-format structural
    // facts (I4 mirror, PARAMETERS now -- the loop body op's pinned attrs): QK_K=256,
    // block_q2_Kx16 weight stride 1344 (16 fp16 d + 16 fp16 dmin + 256 packed 4-bit
    // scale/min + 1024 2-bit quant bytes), block_q8_K activation stride 292, the weight
    // quant bytes at +320, the per-column dmin strip at +32, the packed scale/min
    // region at +64, the activation int8 quants at +4, the activation bsums at +260, 16
    // weight columns per group, 16 sub-blocks of 16, and the VLEN-derived e8 half width.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 8 sub-blocks / super-half

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc, "repack-gemv-q2_K output not pointer");
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
    // vand_vx_u8 / vsrl_vx_u8 -- the 4-bit scale/min unpack and the 2-bit weight
    // lane peel (UNSIGNED, value-identity reinterpret).
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
    // Reinterpret an unsigned 2-bit lane (0..3) to a SIGNED i8 lane (value-identity).
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
    // vwmacc_vx i8->i16: acc += scalar * vec (the 2-bit weight dot chunk).
    std::string vwmacc16Callee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc16 = [&](mlir::Value acc, mlir::Value scalar,
                        mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmacc16Callee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };
    // vwmacc_vv i16->i32: acc += scale_strip(i16) * sumi_s(i16). The per-sub-block
    // 4-bit scale multiplies the per-sub-block i16 partial into the i32 accumulator.
    std::string vwmaccVV32Callee = ("__riscv_vwmacc_vv_i32" + l32).str();
    auto vwmaccVV32 = [&](mlir::Value acc, mlir::Value scaleStrip,
                          mlir::Value sumiS) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVV32Callee,
                            mlir::ValueRange{acc, scaleStrip, sumiS, vl8}, opName,
                            role);
    };
    // vwmacc_vx i16->i32: bsums_acc += bsum_scalar * min_strip(i16). The per-sub-
    // block 4-bit min, weighted by the SINGLE paired activation bsum.
    std::string vwmaccVX32Callee = ("__riscv_vwmacc_vx_i32" + l32).str();
    auto vwmaccVX32 = [&](mlir::Value acc, mlir::Value scalar,
                          mlir::Value minStrip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVX32Callee,
                            mlir::ValueRange{acc, scalar, minStrip, vl8}, opName,
                            role);
    };
    // vzext_vf2 u8 strip -> u16 strip, then reinterpret to i16: the 4-bit scale/min
    // (0..15) lifted to the i16 lane the vwmacc consumes.
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

      // const uint8_t *b = vx + x*nb*1344;  (the q2_Kx16 column group base).
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

        // const uint8_t *bl = b + l*1344;   const uint8_t *al = a + l*292;
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
        // stores a FLOAT (4 bytes) at offset 0. It scales BOTH the main d term and
        // the dmin MIN term.
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

        // -- per-strip dmin_d_h = vfwcvt(vle16(&bl.dmin[h*half])) * d_y (MIN scale).
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

        // ===== Per-block i32 accumulators per strip (scale main + bsums-min). =====
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

        // i16 partial seed for the inner 2-bit dot.
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

        // ===== Super-half loop: for (k = 0; k < QK_K/128; ++k) =====
        // Each super-half is 8 sub-blocks of 16. UNPACK the 8 per-sub-block 4-bit
        // scale strips + 8 min strips LANE-WISE (vand 0x0F scale / vsrl 4 min on the
        // SINGLE packed byte), then the MIN bsums fold, then the 2-bit integer dot.
        for (int64_t k = 0; k < nSuperHalves; ++k) {
          step("scale_min_unpack_superhalf");
          // Per strip h, per sub-block sb (0..7): the packed byte is at
          // scales[k*128 + sb*16 + h*half] (gsub = k*8+sb). scale = byte & 0x0F;
          // min = byte >> 4. Both zero-extended to i16 for the vwmacc chain.
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> scaleVal(numHalves);
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> minVal(numHalves);
          for (int64_t h = 0; h < numHalves; ++h) {
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t gsub = k * subPerSuper + sb;
              int64_t scByte = weightScalesOffset + gsub * 16 + h * half;
              mlir::Value packed = loadU8Strip(bl, sizeLit(scByte));
              mlir::Value scU8 = u8Imm(vandCallee, packed, "0x0F");
              mlir::Value mnU8 = u8Imm(vsrlCallee, packed, "4");
              scaleVal[h].push_back(liftToI16(scU8));
              minVal[h].push_back(liftToI16(mnU8));
            }
          }

          // ----- MIN term: bsums_acc += bsum_sub * min_sub (i32 widen). Each q2_K
          // 16-element sub-block is EXACTLY one q8_K bsums group, so the bsum is a
          // SINGLE int16 read (NOT q4_K's paired bs0+bs1). Global sub = k*8 + sb. --
          step("min_bsums_fold");
          for (int64_t sb = 0; sb < subPerSuper; ++sb) {
            int64_t gsub = k * subPerSuper + sb;
            mlir::Value bsum = i16Read(al, activationBsumsOffset + gsub * 2);
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value curB =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[h])
                      .getResult();
              rewriter.create<emitc::AssignOp>(
                  loc, bsumsVar[h], vwmaccVX32(curB, bsum, minVal[h][sb]));
            }
          }

          // ----- MAIN term: two m-halves of the 32-byte super-half chunk. m-half 0
          // covers the EVEN local sub-blocks (2j+0), m-half 1 the ODD (2j+1). Within
          // an m-half, a 16-position i16 partial per shift-j (sub-block 2j+mh); each
          // qs byte is loaded ONCE per column strip and peeled into 4 2-bit lanes by
          // vsrl {0,2,4,6} + vand 0x03. NO 2x16 k-chunk split (16*127*3 < 32767). --
          for (int64_t mh = 0; mh < 2; ++mh) {
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> sPartial(numHalves);
            if (rolledMainTerm) {
              // ---- ROLLED whole-K-nest main term ([GAP-EMIT-KQUANT-GEVM-TILE-
              // ROUNDTRIP] maturity lever): the dominant per-16-position inner mm-loop is
              // emitted as ONE runtime emitc.for; the per-strip x per-2bit-lane i16
              // partials sPartial[h][j] are carried as RESIDENT SSA-register VariableOps
              // (seeded ABOVE the loop, load-accumulate-store INSIDE it). The per-position
              // qs byte is loaded + peeled ONCE per iteration -- NEVER materialized to a
              // stack scratch tile. BYTE-EXACT to the unrolled emit by construction: the
              // vwmacc16 integer accumulation order (mm ascending, then h, then lane j) is
              // IDENTICAL -- only the loop is materialized instead of unrolled.
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> sPartialVar(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                for (int64_t j = 0; j < 4; ++j) {
                  auto v = rewriter.create<emitc::VariableOp>(
                      loc, emitc::LValueType::get(i16m1Type),
                      emitc::OpaqueAttr::get(ctx, ""));
                  rewriter.create<emitc::AssignOp>(loc, v, seedI16());
                  sPartialVar[h].push_back(v);
                }
              auto mmLoop = rewriter.create<emitc::ForOp>(
                  loc, sizeLit(0), sizeLit(16), sizeLit(1), /*bodyBuilder=*/nullptr);
              {
                mlir::OpBuilder::InsertionGuard ig(rewriter);
                rewriter.setInsertionPointToStart(mmLoop.getBody());
                mlir::Value mmv = mmLoop.getInductionVar();
                // m = mh*16 + mm; runtime weight offset = C + mm*16, runtime activation
                // offset = C + mm (C = the k*32/mh*16-shifted base).
                mlir::Value mmv16 = rewriter.create<emitc::MulOp>(
                    loc, sizeType, mmv, sizeLit(16));
                for (int64_t h = 0; h < numHalves; ++h) {
                  step("weight_2bit_addr");
                  mlir::Value wOff = rewriter.create<emitc::AddOp>(
                      loc, sizeType,
                      sizeLit(weightQuantOffset + (k * 32 + mh * 16) * 16 + h * half),
                      mmv16);
                  mlir::Value packed = loadU8Strip(bl, wOff);
                  for (int64_t j = 0; j < 4; ++j) {
                    mlir::Value shifted = packed;
                    if (j != 0)
                      shifted = u8Imm(vsrlCallee, packed, std::to_string(2 * j));
                    mlir::Value w =
                        reinterpretToI8(u8Imm(vandCallee, shifted, "0x03"));
                    step("act_quant_addr");
                    mlir::Value aqOff = rewriter.create<emitc::AddOp>(
                        loc, sizeType,
                        sizeLit(activationQuantOffset + k * 128 + j * 32 + mh * 16),
                        mmv);
                    mlir::Value aq = i8Read(al, aqOff);
                    mlir::Value cur =
                        rewriter.create<emitc::LoadOp>(loc, i16m1Type,
                                                       sPartialVar[h][j])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(loc, sPartialVar[h][j],
                                                     vwmacc16(cur, aq, w));
                  }
                }
              }
              for (int64_t h = 0; h < numHalves; ++h)
                for (int64_t j = 0; j < 4; ++j)
                  sPartial[h].push_back(
                      rewriter.create<emitc::LoadOp>(loc, i16m1Type,
                                                     sPartialVar[h][j])
                          .getResult());
            } else {
              // ---- UNROLLED main term (default, register-resident full unroll) ----
              for (int64_t h = 0; h < numHalves; ++h)
                for (int64_t j = 0; j < 4; ++j)
                  sPartial[h].push_back(seedI16());
              for (int64_t mm = 0; mm < 16; ++mm) {
                int64_t m = mh * 16 + mm;
                for (int64_t h = 0; h < numHalves; ++h) {
                  step("weight_2bit_addr");
                  // qs byte index within the super-block = k*32 + m.
                  mlir::Value packed = loadU8Strip(
                      bl, sizeLit(weightQuantOffset + (k * 32 + m) * 16 + h * half));
                  for (int64_t j = 0; j < 4; ++j) {
                    // 2-bit lane j = (byte >> 2j) & 0x03.
                    mlir::Value shifted = packed;
                    if (j != 0)
                      shifted = u8Imm(vsrlCallee, packed, std::to_string(2 * j));
                    mlir::Value w =
                        reinterpretToI8(u8Imm(vandCallee, shifted, "0x03"));
                    step("act_quant_addr");
                    // activation global position = k*128 + j*32 + m.
                    mlir::Value aq = i8Read(
                        al, sizeLit(activationQuantOffset + k * 128 + j * 32 + m));
                    sPartial[h][j] = vwmacc16(sPartial[h][j], aq, w);
                  }
                }
              }
            }
            // sumi += scale_{2j+mh} * partial_j (i16->i32 vwmacc_vv).
            step("scale_subblock_fold");
            for (int64_t h = 0; h < numHalves; ++h) {
              for (int64_t j = 0; j < 4; ++j) {
                mlir::Value cur =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                        .getResult();
                rewriter.create<emitc::AssignOp>(
                    loc, sumiVar[h],
                    vwmaccVV32(cur, scaleVal[h][2 * j + mh], sPartial[h][j]));
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
    (void)i32m1Type;
    // RESULT-LESS (no monolith token): the typed_repack_gemv_loop_body region is
    // result-less (the repacked lane-wise K-quant GEVM sinks through the output
    // pointer, not an SSA vector), so unlike the retired direct emitter this body leaf
    // seeds NO dead i32m1 result token.
    return mlir::success();
  }

// q2_K x q8_K 16x1-REPACKED PREFILL GEMM (prefill) body leaf, from the FRONT DOOR
// + S6 TILED (byte-exact). The q2_K prefill sibling of emitRepackKQuantGemvBodyQ2K:
// the SAME 2-bit-4-lanes weight assembly + 4-bit packed scale/min + dual d/dmin +
// bsums-min fold, with the WEIGHT-side 2-bit decode and scale/min unpack done ONCE
// per 16-weight group and REUSED across the 4 interleaved activation columns of
// block_q8_Kx4 -- the amortization the single-column GEVM lacks. Because q2_K SHARES
// the q4_K dual d/dmin + bsums-min fold, the FAMILY S6 tiling scheme transfers WHOLE
// (the register-cliff lever): S1 h-strip OUTPUT TILE (one disjoint weight strip at a
// time, a SEPARATE emitc.for per strip scoping each block loop's live set) + the S6
// stack panels -- the decoded 4-bit scale/min i16 strips are staged to scalePanel/
// minPanel and the idle per-column i32 MIN accumulator is staged to bsumsPanel (both
// addressed via vle/vse opaque intrinsics so they LIVE ON THE STACK, off the hot live
// set) + on-demand d/dmin widen (deferred to the end-of-block fold). The HOT sumf/sumi
// per-column accumulators stay SSA-register VariableOps (NEVER paneled, NEVER rolled
// into the iter-arg-less emitc.for -- the re-roll trap). Every staged value is INTEGER
// (exact store/reload) and the end-of-block f32 fold order is UNCHANGED => byte-exact
// bit-identical to the plain untiled emit. Activation interleave (IDENTICAL to q4_K):
// qs@16 are 4-column-interleaved (element e of column c at qs[e*4 + c]); bsums@1040
// are group16-major/column-minor (group g16 col c at bsums[g16*4 + c]); d[4]@0 are 4
// fp32 scalars. The block-format facts are PARAMETERS (the loop body op's pinned
// attrs, read by the K-quant branch of emitTypedRepackGemmLoopBody and passed in).
// NUMERIC STATUS: oracle-verified byte-exact INTEGER (isum + summs) vs the INDEPENDENT
// scalar q2_K dequant-matmul reference on the interleaved x4 stream. RESULT-LESS.
mlir::LogicalResult VariantToEmitCFunc::emitRepackKQuantGemmBodyQ2K(
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

    // The integer-product core LMUL anchor (the *how*, never the *what*), the SAME
    // parametric chain the q2_K GEVM sibling carries.
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
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
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

    // The 16x1 repacked q2_K PREFILL GEMM block-format facts (I4 mirror, PARAMETERS
    // now): QK_K=256, block_q2_Kx16 weight stride 1344 (SAME weight ABI as the GEVM),
    // block_q8_Kx4 activation stride 1168 (4 fp32 d + 1024 int8 quants [4 columns
    // interleaved] + 64 int16 bsums), weight quants at +320, dmin strip at +32, packed
    // scale/min region at +64, interleaved activation quants at +16, interleaved
    // activation bsums at +1040, 16 weight / 4 activation columns per group, 16
    // sub-blocks of 16, and the VLEN-derived e8 half width.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 8 sub-blocks / super-half
    // RVV1.0 folds all 4 activation columns in one pass; RVV0.7.1 (whole-LMUL) folds
    // ONE column per pass to bound the live set (the q4_K/q4_1 GEMM spill rationale).
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave;  // 1 @rvv07; 4 @rvv1.0

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-q2_K output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

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
    std::string vwmaccVX32Callee = ("__riscv_vwmacc_vx_i32" + l32).str();
    auto vwmaccVX32 = [&](mlir::Value acc, mlir::Value scalar,
                          mlir::Value minStrip) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwmaccVX32Callee,
                            mlir::ValueRange{acc, scalar, minStrip, vl8}, opName,
                            role);
    };
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

        // Activation-column-PASS loop (compile-time): the columns [cLo, cHi) folded
        // in this pass over the block loop.
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
        // ===== S1 h-strip OUTPUT TILE (mr=4 / hs=1) ======================
        // Process one DISJOINT weight strip h at a time, each in its OWN contraction-
        // block loop (a SEPARATE emitc.for scopes each block loop's live set), so the
        // register allocator only ever juggles ONE strip's accumulators + decode
        // temporaries at once (the q4_K spill-cliff lever, transferred WHOLE since
        // q2_K shares the min fold). The per-column f32 accumulator sumf_c is TILE-
        // LOCAL: columnsPerPass accumulators seeded ABOVE this strip's block loop and
        // carried across it as SSA-register VariableOps (NEVER rolled into the
        // iter-arg-less emitc.for). Amortization is UNCHANGED: the h strips read
        // DISJOINT weight bytes, and the weight decode stays amortized across the 4
        // activation columns WITHIN each strip. numHalves == weight_interleave /
        // half_lanes is the VLEN-derived tile count (2 @VLEN128, 1 @VLEN256).
        for (int64_t h = 0; h < numHalves; ++h) {
        // sumfVar[c]: THIS strip's per-column f32 accumulator, carried across the
        // contraction-block loop. The MIN correction is folded straight into sumf via
        // vfnmsac at end-of-block.
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

          // const uint8_t *bl = b + l*1344;   const uint8_t *al = a + l*1168;
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

          // ===== S6 decoded-weight + min-accumulator STACK PANELS (byte-exact) ===
          // The COLD decode band (this block's 4-bit scale/min i16 strips) and the
          // COLD per-column i32 MIN accumulator are staged into small PER-BLOCK stack
          // arrays, addressed via vle/vse opaque intrinsics (the taken address defeats
          // mem2reg, so they LIVE ON THE STACK, not in vregs). This takes the scale
          // strips (held live across the dot loop only for the post-loop fold) and the
          // bsums family (idle throughout the hot main term) OFF the hot live set, so
          // the allocator only juggles the HOT sumf/sumi accumulators + sPartial
          // partials. Every staged value is INTEGER (exact store/reload) and the
          // end-of-block f32 fold is UNCHANGED => byte-exact. sumf/sumi stay SSA-
          // register accumulators (the re-roll trap avoided). Declared PER-BLOCK.
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
          // scale/min panels are per-super-half local (subPerSuper strips), rewritten
          // each super-half; bsums panel is per-block (columnsPerPass).
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

          // -- S6 d/dmin ON-DEMAND: the per-column-lane fp16 d/dmin strips of THIS h
          // are per-block loop-invariant but are only CONSUMED in the end-of-block f32
          // fold. Their widen-to-f32 is deferred to that fold point (see below), so
          // the two f32m2 strips do NOT sit live across the hot main-term dot loop.

          // ===== Per-column i32 main accumulator for THIS strip (SSA register). =====
          // sumiVar[c] (scale main term) stays a HOT SSA-register accumulator; the
          // bsums (min term) accumulator family is STAGED to bsumsPanel -- idle across
          // the hot main dot and reloaded only at the end-of-block fold.
          llvm::SmallVector<mlir::Value> sumiVar(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            auto sv = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(i32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
            sumiVar[c] = sv;
          }

          // ===== Super-half loop: for (k = 0; k < QK_K/128; ++k) =====
          for (int64_t k = 0; k < nSuperHalves; ++k) {
            step("scale_min_unpack_superhalf");
            // Per sub-block sb (0..subPerSuper): the packed byte is at
            // scales[k*128 + sb*16 + h*half] (gsub = k*subPerSuper + sb). scale = byte
            // & 0x0F; min = byte >> 4. STAGED to scalePanel/minPanel (byte-exact i16)
            // and reloaded per column in the MIN/MAIN folds.
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t gsub = k * subPerSuper + sb;
              int64_t scByte = weightScalesOffset + gsub * 16 + h * half;
              mlir::Value packed = loadU8Strip(bl, sizeLit(scByte));
              mlir::Value scU8 = u8Imm(vandCallee, packed, "0x0F");
              mlir::Value mnU8 = u8Imm(vsrlCallee, packed, "4");
              storeI16Panel(scalePanel, sb * half, liftToI16(scU8));
              storeI16Panel(minPanel, sb * half, liftToI16(mnU8));
            }

            // ----- MIN term per activation column c: bsums_acc[c] += bsum * min_sb.
            // The interleaved q8_Kx4 bsums are group16-major/column-minor (index =
            // group16*4 + c); q2_K sub-block gsub IS group16 gsub, so the bsum is a
            // SINGLE read a.bsums[gsub*4 + c] (NOT q4_K's paired read). The min strip
            // is reloaded ONCE per sub-block from the panel, reused across columns; the
            // bsums accumulator is read/updated straight in the panel (byte-exact i32).
            step("min_bsums_fold");
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t gsub = k * subPerSuper + sb;
              mlir::Value minStrip = loadI16Panel(minPanel, sb * half);
              for (int64_t c = cLo; c < cHi; ++c) {
                mlir::Value bsum = i16Read(
                    al, activationBsumsOffset + (gsub * 4 + c) * 2);
                // First sub-block (k==0 && sb==0) DEFINES the panel from a register
                // zero seed (no zero-store); later sub-blocks load-accumulate-store.
                bool firstAcc = (k == 0 && sb == 0);
                mlir::Value cur =
                    firstAcc ? seedI32() : loadBsumsPanel(c - cLo);
                storeBsumsPanel(c - cLo, vwmaccVX32(cur, bsum, minStrip));
              }
            }

            // ----- MAIN term: two m-halves; the weight 2-bit lanes are DECODED ONCE
            // per (m) for THIS strip and REUSED across the activation columns, each
            // column reading its own interleaved q8_Kx4 quant qs[16 + (k*128+j*32+m)*4
            // + c]. NO 2x16 k-chunk split (16*127*3 < 32767). -----
            for (int64_t mh = 0; mh < 2; ++mh) {
              // Slot (c,j) of THIS strip's per-column-per-shift i16 partial family.
              auto partIdx = [&](int64_t c, int64_t j) -> int64_t {
                return ((c - cLo) * 4 + j) * half;
              };
              if (rolledMainTerm) {
                // ---- ROLLED main term ([GAP-EMIT-UNROLL] maturity lever) --------
                // The per-16-weight-group inner loop is emitted as ONE runtime
                // emitc.for (compact code volume) INSTEAD of the register-resident
                // full static unroll; the per-column-per-shift i16 partials are staged
                // to a PER-mh stack panel, load-accumulate-store per iteration. This
                // trades register residence for code compactness (the capability-keyed
                // schedule tradeoff). BYTE-EXACT to the unrolled emit by construction:
                // the vwmacc16 integer accumulation order (mm ascending, then j, then
                // c) is IDENTICAL -- only the loop is materialized instead of unrolled,
                // and integer add is order-exact regardless. The end-of-block f32 fold
                // (below) is UNCHANGED.
                mlir::Type sPartialPanelTy = emitc::ArrayType::get(
                    {columnsPerPass * 4 * half}, i16ElemTy);
                mlir::TypedValue<emitc::ArrayType> sPartialPanel =
                    mkPanel(sPartialPanelTy);
                for (int64_t c = cLo; c < cHi; ++c)
                  for (int64_t j = 0; j < 4; ++j)
                    storeI16Panel(sPartialPanel, partIdx(c, j), seedI16());

                auto mmLoop = rewriter.create<emitc::ForOp>(
                    loc, sizeLit(0), sizeLit(16), sizeLit(1),
                    /*bodyBuilder=*/nullptr);
                {
                  mlir::OpBuilder::InsertionGuard mg(rewriter);
                  rewriter.setInsertionPointToStart(mmLoop.getBody());
                  mlir::Value mmv = mmLoop.getInductionVar();
                  // SHARED weight 2-bit decode per m (THIS strip): 4 lanes reused over
                  // columns. Runtime weight byte offset = C0w + mm*16.
                  step("weight_2bit_addr");
                  int64_t c0w =
                      weightQuantOffset + (k * 32 + mh * 16) * 16 + h * half;
                  mlir::Value mmv16 =
                      rewriter.create<emitc::MulOp>(loc, sizeType, mmv,
                                                    sizeLit(16));
                  mlir::Value wOff = rewriter.create<emitc::AddOp>(
                      loc, sizeType, sizeLit(c0w), mmv16);
                  mlir::Value packed = loadU8Strip(bl, wOff);
                  llvm::SmallVector<mlir::Value> wLane;
                  for (int64_t j = 0; j < 4; ++j) {
                    mlir::Value shifted = packed;
                    if (j != 0)
                      shifted = u8Imm(vsrlCallee, packed, std::to_string(2 * j));
                    wLane.push_back(
                        reinterpretToI8(u8Imm(vandCallee, shifted, "0x03")));
                  }
                  // Runtime activation byte offset per (j,c) = C1 + mm*4.
                  mlir::Value mmv4 = rewriter.create<emitc::MulOp>(
                      loc, sizeType, mmv, sizeLit(4));
                  for (int64_t c = cLo; c < cHi; ++c) {
                    for (int64_t j = 0; j < 4; ++j) {
                      step("act_quant_addr");
                      int64_t c1 = activationQuantOffset +
                                   (k * 128 + j * 32 + mh * 16) * 4 + c;
                      mlir::Value aOff = rewriter.create<emitc::AddOp>(
                          loc, sizeType, sizeLit(c1), mmv4);
                      mlir::Value aq = i8Read(al, aOff);
                      mlir::Value cur = loadI16Panel(sPartialPanel, partIdx(c, j));
                      storeI16Panel(sPartialPanel, partIdx(c, j),
                                    vwmacc16(cur, aq, wLane[j]));
                    }
                  }
                }
                // sumi_c += scale_{2j+mh} * partial_{c,j} -- the partials are read from
                // the panel (final accumulated i16), the fold order (j ascending, c) is
                // unchanged => byte-exact.
                step("scale_subblock_fold");
                for (int64_t j = 0; j < 4; ++j) {
                  mlir::Value scStrip =
                      loadI16Panel(scalePanel, (2 * j + mh) * half);
                  for (int64_t c = cLo; c < cHi; ++c) {
                    mlir::Value part = loadI16Panel(sPartialPanel, partIdx(c, j));
                    mlir::Value cur =
                        rewriter
                            .create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(
                        loc, sumiVar[c], vwmaccVV32(cur, scStrip, part));
                  }
                }
                continue; // next m-half
              }
              // ---- UNROLLED main term (default, register-resident full unroll) ----
              // Per-column i16 partials, one per shift-j (THIS strip).
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> sPartial(
                  activationInterleave);
              for (int64_t c = cLo; c < cHi; ++c)
                for (int64_t j = 0; j < 4; ++j)
                  sPartial[c].push_back(seedI16());
              for (int64_t mm = 0; mm < 16; ++mm) {
                int64_t m = mh * 16 + mm;
                // SHARED weight 2-bit decode per m (THIS strip): 4 lanes reused over
                // columns. wLane[j].
                step("weight_2bit_addr");
                mlir::Value packed = loadU8Strip(
                    bl, sizeLit(weightQuantOffset + (k * 32 + m) * 16 + h * half));
                llvm::SmallVector<mlir::Value> wLane;
                for (int64_t j = 0; j < 4; ++j) {
                  mlir::Value shifted = packed;
                  if (j != 0)
                    shifted = u8Imm(vsrlCallee, packed, std::to_string(2 * j));
                  wLane.push_back(
                      reinterpretToI8(u8Imm(vandCallee, shifted, "0x03")));
                }
                for (int64_t c = cLo; c < cHi; ++c) {
                  for (int64_t j = 0; j < 4; ++j) {
                    step("act_quant_addr");
                    mlir::Value aq = i8Read(
                        al, sizeLit(activationQuantOffset +
                                    (k * 128 + j * 32 + m) * 4 + c));
                    sPartial[c][j] = vwmacc16(sPartial[c][j], aq, wLane[j]);
                  }
                }
              }
              // sumi_c += scale_{2j+mh} * partial_{c,j} (i16->i32 vwmacc_vv). The scale
              // strip is reloaded ONCE per j from the panel, reused across the columns
              // (byte-exact i16); the per-column accumulation order (j ascending) is
              // unchanged.
              step("scale_subblock_fold");
              for (int64_t j = 0; j < 4; ++j) {
                mlir::Value scStrip =
                    loadI16Panel(scalePanel, (2 * j + mh) * half);
                for (int64_t c = cLo; c < cHi; ++c) {
                  mlir::Value cur =
                      rewriter
                          .create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c], vwmaccVV32(cur, scStrip, sPartial[c][j]));
                }
              }
            }
          }

          // ===== End-of-block fold per column for THIS strip: sumf_c += d_x*d_y_c*sumi
          // (main) then sumf_c -= dmin_x*d_y_c*bsums_c (MIN). =====
          // S6: the d/dmin widen is loaded HERE (on-demand at the fold), not at the top
          // of the block, so it does not occupy 4 vreg across the main dot loop.
          mlir::Value dminF32 =
              widenF16(loadF16Strip(bl, weightDminOffset, h * half));
          mlir::Value dF32 = widenF16(loadF16Strip(bl, 0, h * half));
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
            // dmin_0_c = dminF32 * d_y_c;  sumf_c -= dmin_0_c * cvt(bsums_c).
            // bsums_c reloaded from its stack panel (byte-exact i32); the fold order
            // (main vfmacc THEN min vfnmsac) is identical to the plain emit ->
            // bit-identical.
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
    // RESULT-LESS (no monolith token): the typed_repack_gemm_loop_body region is
    // result-less (the repacked lane-wise K-quant GEMM sinks through the output
    // pointer, not an SSA vector), so unlike the retired direct emitter this body leaf
    // seeds NO dead i32m1 result token.
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
