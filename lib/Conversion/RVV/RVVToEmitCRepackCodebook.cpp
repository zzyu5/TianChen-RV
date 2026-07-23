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

// Typed codebook repack artifact consumers.

mlir::LogicalResult VariantToEmitCFunc::emitRepackCodebookGemvBodyIq4Nl(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t activationQuantOffset,
    llvm::ArrayRef<int8_t> codebook, int64_t weightInterleave,
    int64_t half) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*). "mf2"
    // (default) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2
    // (f16 scale m1). "m1" is the RVV0.7.1 whole-LMUL chain i8m1 -> i16m2 -> i32m4.
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

    // block_iq4_nlx16 repack facts (I4 mirror) are PARAMETERS now (the loop body op's
    // pinned attrs + the codebook core brick's kvalues, read by the codebook branch of
    // emitTypedRepackGemvLoopBody and passed in): qk, weightStride, activationStride,
    // weightQuantOffset, activationQuantOffset, codebook, weightInterleave, half.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nibbleBytes = qk / 2;                    // 16 nibble bytes
    int64_t activationHighRow = nibbleBytes;         // 16 (high nibble -> pos i+16)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(
          loc, "repack-gemv-iq4_nl output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The 16-entry non-linear int8 codebook is a STRUCTURAL fact off the typed
    // attr (I4 mirror). Emit it ONCE as a `static const int8_t[16]` decl; the
    // MEMORY codebook GATHER (vluxei16) indexes it by nibble byte offset.
    llvm::StringLiteral tableName = "weft_iq4_nl_repack_kvalues";
    {
      std::string decl =
          ("static const int8_t " + tableName + "[16] = {").str();
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load / codebook-gather helpers ----
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadNibbles = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
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
    // The REAL codebook GATHER: zero-extend the nibble index (0..15) to a u16
    // byte offset (into the int8[16] codebook, so offset == index), then
    // vluxei16_v_i8 gathers codebook[nibble] per lane. This is the fractional-
    // anchor MEMORY gather (a register vrgather needs VLMAX >= 16 to index all 16
    // entries, illegal at the mf2 anchor's 8-lane VLMAX). NOT fake-linear.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto codebookGather = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value off = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("codebook_index_offset"));
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value tablePtr =
                rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName);
            return {tablePtr, off, vl8};
          },
          llvm::StringRef("codebook_gather"));
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
    // vwmul_vx i8->i16 product, then vwadd_wv i16->i32 accumulate (codebook
    // products reach 127*127=16129, so a single product fits i16 but accumulation
    // must be i32).
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };

    mlir::Value aBase = activationBase;

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
            loc, activationPtrType, aBase, alOff);

        // Per-strip i32 accumulator (single, no min).
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
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
        for (int64_t i = 0; i < nibbleBytes; ++i) {
          for (int64_t h = 0; h < numHalves; ++h) {
            step("weight_nibble_addr");
            mlir::Value packed =
                loadNibbles(bl, weightQuantOffset + i * 16 + h * half);
            mlir::Value wLo =
                codebookGather(u8Imm(vandCallee, packed, "0x0F"));
            mlir::Value wHi = codebookGather(u8Imm(vsrlCallee, packed, "0x04"));
            step("act_quant_addr");
            mlir::Value aLo = i8Read(al, activationQuantOffset + i);
            mlir::Value aHi =
                i8Read(al, activationQuantOffset + activationHighRow + i);
            mlir::Value curLo =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                    .getResult();
            mlir::Value afterLo = vwaddw(curLo, vwmul(aLo, wLo));
            rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                             vwaddw(afterLo, vwmul(aHi, wHi)));
          }
        }

        // ===== End-of-block SINGLE-scale fold: sumf += cvt(sumi) * (d_x*d_y). ==
        std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
        auto loadScales = [&](int64_t laneOff) -> mlir::Value {
          step("weight_scale_addr");
          mlir::Value dFull = bl;
          if (laneOff != 0)
            dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                                  sizeLit(laneOff * 2));
          mlir::Value dCast =
              rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
          return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                                mlir::ValueRange{dCast, vl8}, opName, role);
        };
        mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
        llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
        mlir::Value aD = emitOpaqueCallBuilt(
            rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, al)
                      .getResult();
              return {aDCast};
            },
            llvm::StringRef("act_scale_scalar"));
        std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value bD = loadScales(h * half);
          mlir::Value dC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bD, aD, vl8}, opName, role);
          mlir::Value sumiV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult();
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumiV, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                             mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], nextF);
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

    // The typed_repack_gemv_loop_body region op is RESULT-LESS (the per-strip lane-wise
    // vse32 is the sink), so unlike the retired monolith direct emitter there is NO
    // trailing unused-result token to seed.
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitRepackCodebookGemmBodyIq4Nl(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQuantOffset,
    int64_t activationQuantOffset, llvm::ArrayRef<int8_t> codebook,
    int64_t weightInterleave, int64_t activationInterleave,
    int64_t half, bool colGroupOuter) const {
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

    // block_iq4_nlx16 / block_q8_0x4 repack facts (I4 mirror) are PARAMETERS now (the loop
    // body op's pinned attrs + the codebook core brick's kvalues, read by the codebook
    // branch of emitTypedRepackGemmLoopBody and passed in).
    int64_t numHalves = weightInterleave / half;
    int64_t nibbleBytes = qk / 2;                    // 16
    int64_t activationHighRow = nibbleBytes;         // 16
    int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(
          loc, "repack-gemm-iq4_nl output not pointer");

    mlir::Value vl8 = sizeLit(half);

    llvm::StringLiteral tableName = "weft_iq4_nl_repack_kvalues";
    {
      std::string decl =
          ("static const int8_t " + tableName + "[16] = {").str();
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

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
    auto loadNibbles = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
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
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto codebookGather = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value off = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("codebook_index_offset"));
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value tablePtr =
                rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName);
            return {tablePtr, off, vl8};
          },
          llvm::StringRef("codebook_gather"));
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
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadScales = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      if (laneOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(laneOff * 2));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
    llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
    std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");

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

            // Per-column activation scale d_y_c (4 fp16 at 0,2,4,6).
            llvm::SmallVector<mlir::Value> aD(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c) {
              aD[c] = emitOpaqueCallBuilt(
                  rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value aDPtr = al;
                    if (c != 0)
                      aDPtr = rewriter.create<emitc::AddOp>(
                          loc, activationPtrType, al, sizeLit(c * 2));
                    mlir::Value aDCast =
                        rewriter.create<emitc::CastOp>(loc, f16PtrType, aDPtr)
                            .getResult();
                    return {aDCast};
                  },
                  llvm::StringRef("act_scale_scalar"));
            }

            // SHARED weight d strip (per-column-lane fp16).
            llvm::SmallVector<mlir::Value> bD(numHalves);
            for (int64_t h = 0; h < numHalves; ++h)
              bD[h] = loadScales(bl, h * half);

            // Per-column i32 accumulator (single, no min).
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

            // ===== Nibble-step loop: SHARED codebook decode, per-column dot. ===
            for (int64_t i = 0; i < nibbleBytes; ++i) {
              llvm::SmallVector<mlir::Value> wLo(numHalves), wHi(numHalves);
              for (int64_t h = 0; h < numHalves; ++h) {
                step("weight_nibble_addr");
                mlir::Value packed =
                    loadNibbles(bl, weightQuantOffset + i * 16 + h * half);
                wLo[h] = codebookGather(u8Imm(vandCallee, packed, "0x0F"));
                wHi[h] = codebookGather(u8Imm(vsrlCallee, packed, "0x04"));
              }
              for (int64_t c = cLo; c < cHi; ++c) {
                step("act_quant_addr");
                mlir::Value aLo = i8Read(
                    al, activationQuantOffset + i * activationInterleave + c);
                mlir::Value aHi =
                    i8Read(al, activationQuantOffset +
                                   (i + activationHighRow) * activationInterleave +
                                   c);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value cur =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     sumiVar[c][h])
                          .getResult();
                  mlir::Value afterLo = vwaddw(cur, vwmul(aLo, wLo[h]));
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c][h], vwaddw(afterLo, vwmul(aHi, wHi[h])));
                }
              }
            }

            // ===== End-of-block per-column single-scale fold. =====
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value dC =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                                   mlir::ValueRange{bD[h], aD[c], vl8}, opName,
                                   role);
                mlir::Value sumiV =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                        .getResult();
                mlir::Value sumiF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumiV, vl8}, opName, role);
                mlir::Value curF =
                    rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                        .getResult();
                mlir::Value nextF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                                   mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                                   role);
                rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], nextF);
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

    // The typed_repack_gemm_loop_body region op is RESULT-LESS (the per-column per-strip
    // lane-wise vse32 is the sink), so unlike the retired monolith direct emitter there is
    // NO trailing unused-result token to seed.
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitRepackCodebookGemvBodyMxfp4(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t activationQuantOffset,
    llvm::ArrayRef<int8_t> codebook, int64_t weightInterleave,
    int64_t half) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*). "mf2"
    // (default) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2.
    // "m1" is the RVV0.7.1 whole-LMUL chain i8m1 -> i16m2 -> i32m4.
    llvm::StringRef l8 = coreLmul;
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";
    // The E8M0 bit-construction runs in the u32 domain at the f32-accumulator LMUL
    // (u8 mf2 -> u32 m2 via vf4). The compare-mask ratio is SEW/LMUL: u32 m2 -> b16,
    // u32 m4 -> b8.
    llvm::StringRef maskN = l32 == "m4" ? "8" : "16";
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    mlir::Type u32m2Type =
        emitc::OpaqueType::get(ctx, ("vuint32" + l32 + "_t").str());
    mlir::Type boolMaskType =
        emitc::OpaqueType::get(ctx, ("vbool" + maskN + "_t").str());
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

    // block_mxfp4x16 repack facts (I4 mirror) are PARAMETERS now (the loop body op's
    // pinned attrs + the codebook core brick's kvalues, read by the codebook branch of
    // emitTypedRepackGemvLoopBody and passed in): qk, weightStride, activationStride,
    // weightQuantOffset, activationQuantOffset, codebook, weightInterleave, half.
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t nibbleBytes = qk / 2;                    // 16 nibble bytes
    int64_t activationHighRow = nibbleBytes;         // 16 (high nibble -> pos i+16)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemv-mxfp4 output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The 16-entry doubled-E2M1 int8 fp4 codebook is a STRUCTURAL fact off the typed
    // attr (I4 mirror). Emit it ONCE as a `static const int8_t[16]` decl; the MEMORY
    // codebook GATHER (vluxei16) indexes it by nibble byte offset.
    llvm::StringLiteral tableName = "weft_mxfp4_repack_kvalues";
    {
      std::string decl =
          ("static const int8_t " + tableName + "[16] = {").str();
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load / codebook-gather helpers (SHARED with iq4_nl) ----
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadNibbles = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
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
    // The REAL fp4-codebook GATHER: zero-extend the nibble index (0..15) to a u16
    // byte offset (into the int8[16] codebook, so offset == index), then
    // vluxei16_v_i8 gathers kvalues_mxfp4[nibble] per lane. The fractional-anchor
    // MEMORY gather (a register vrgather needs VLMAX >= 16, illegal at the mf2
    // anchor's 8-lane VLMAX). NOT fake-linear -- IDENTICAL to iq4_nl.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto codebookGather = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value off = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("codebook_index_offset"));
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value tablePtr =
                rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName);
            return {tablePtr, off, vl8};
          },
          llvm::StringRef("codebook_gather"));
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
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };

    // ---- The E8M0 shared-exponent WEIGHT-SCALE strip (the mxfp4-distinguishing
    // fact). Load `half` E8M0 exponent bytes (one per column-lane) at bl + laneOff
    // (the E8M0 region is at weight offset +0, 1 byte per column), then reconstruct
    // 2^(e-128) per lane by ggml's EXACT bit construction:
    //   e32   = (uint32_t) e                                   (vzext_vf4)
    //   bits  = (e < 2) ? (0x00200000u << (e & 0x1F)) : ((e-1) << 23)
    //   scale = *(float *)&bits                                (vreinterpret)
    // All lane-wise (vsll_vv denorm shift / vsll_vx normal shift / vmerge select),
    // the vectorized sibling of the scalar emitE8M0HalfScale. Returns a f32 vector.
    std::string vzextVf4Callee = ("__riscv_vzext_vf4_u32" + l32).str();
    std::string vandU32Callee = ("__riscv_vand_vx_u32" + l32).str();
    std::string vmvU32Callee = ("__riscv_vmv_v_x_u32" + l32).str();
    std::string vsllVvU32Callee = ("__riscv_vsll_vv_u32" + l32).str();
    std::string vsubU32Callee = ("__riscv_vsub_vx_u32" + l32).str();
    std::string vsllVxU32Callee = ("__riscv_vsll_vx_u32" + l32).str();
    std::string vmsltuCallee =
        ("__riscv_vmsltu_vx_u32" + l32 + "_b" + maskN).str();
    std::string vmergeU32Callee = ("__riscv_vmerge_vvm_u32" + l32).str();
    std::string vreinterpretCallee =
        ("__riscv_vreinterpret_v_u32" + l32 + "_f32" + l32).str();
    auto u32Imm = [&](llvm::StringRef callee, mlir::Type resTy, mlir::Value v,
                      llvm::StringRef imm) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, resTy, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, imm)
                    .getResult();
            return {v, immV, vl8};
          });
    };
    auto e8m0ScaleStrip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_e8m0_scale_addr");
      mlir::Value eFull = bl;
      if (laneOff != 0)
        eFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(laneOff));
      mlir::Value eCast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, eFull).getResult();
      mlir::Value eU8 = emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                                       mlir::ValueRange{eCast, vl8}, opName, role,
                                       llvm::StringRef("e8m0_exponent_load"));
      mlir::Value e32 =
          emitOpaqueCall(rewriter, loc, u32m2Type, vzextVf4Callee,
                         mlir::ValueRange{eU8, vl8}, opName, role,
                         llvm::StringRef("e8m0_widen_u32"));
      step("e8m0_to_fp32_half_bits");
      // denorm branch: 0x00200000u << (e & 0x1F).
      mlir::Value emask = u32Imm(vandU32Callee, u32m2Type, e32, "0x1F");
      mlir::Value base = emitOpaqueCallBuilt(
          rewriter, loc, u32m2Type, vmvU32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value lit = rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                                "0x00200000")
                                  .getResult();
            return {lit, vl8};
          });
      mlir::Value denorm =
          emitOpaqueCall(rewriter, loc, u32m2Type, vsllVvU32Callee,
                         mlir::ValueRange{base, emask, vl8}, opName, role);
      // normal branch: (e - 1) << 23.
      mlir::Value em1 = u32Imm(vsubU32Callee, u32m2Type, e32, "1");
      mlir::Value norm = u32Imm(vsllVxU32Callee, u32m2Type, em1, "23");
      // select mask e < 2, then merge (denorm when mask true).
      mlir::Value mask = u32Imm(vmsltuCallee, boolMaskType, e32, "2");
      mlir::Value bits =
          emitOpaqueCall(rewriter, loc, u32m2Type, vmergeU32Callee,
                         mlir::ValueRange{norm, denorm, mask, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, f32m2Type, vreinterpretCallee,
                            mlir::ValueRange{bits}, opName, role,
                            llvm::StringRef("e8m0_reinterpret_f32"));
    };

    mlir::Value aBase = activationBase;

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
            loc, activationPtrType, aBase, alOff);

        // Per-strip i32 accumulator (single, no min).
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
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
        for (int64_t i = 0; i < nibbleBytes; ++i) {
          for (int64_t h = 0; h < numHalves; ++h) {
            step("weight_nibble_addr");
            mlir::Value packed =
                loadNibbles(bl, weightQuantOffset + i * 16 + h * half);
            mlir::Value wLo =
                codebookGather(u8Imm(vandCallee, packed, "0x0F"));
            mlir::Value wHi = codebookGather(u8Imm(vsrlCallee, packed, "0x04"));
            step("act_quant_addr");
            mlir::Value aLo = i8Read(al, activationQuantOffset + i);
            mlir::Value aHi =
                i8Read(al, activationQuantOffset + activationHighRow + i);
            mlir::Value curLo =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                    .getResult();
            mlir::Value afterLo = vwaddw(curLo, vwmul(aLo, wLo));
            rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                             vwaddw(afterLo, vwmul(aHi, wHi)));
          }
        }

        // ===== End-of-block SINGLE-scale fold: sumf += cvt(sumi) * (scale_x*d_y).
        // scale_x is the E8M0-reconstructed per-column f32 weight-scale vector;
        // d_y is the plain block_q8_0 fp16 activation delta (cast to float). ====
        mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
        llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
        mlir::Value aD = emitOpaqueCallBuilt(
            rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, al)
                      .getResult();
              return {aDCast};
            },
            llvm::StringRef("act_scale_scalar"));
        mlir::Value aDf =
            rewriter.create<emitc::CastOp>(loc, floatType, aD).getResult();
        std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value scaleX = e8m0ScaleStrip(bl, h * half);
          mlir::Value dC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                             mlir::ValueRange{scaleX, aDf, vl8}, opName, role);
          mlir::Value sumiV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult();
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumiV, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                             mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], nextF);
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

    // The typed_repack_gemv_loop_body region op is RESULT-LESS (the per-strip lane-wise
    // vse32 is the sink), so unlike the retired monolith direct emitter there is NO
    // trailing unused-result token to seed.
    return mlir::success();
  }

// NOTE (G3 retirement_batch 4): the mxfp4 repack GEVM direct emitter emitRepackGemvMxfp4Q8
// (+ its transitional thin shim) is RETIRED with the GgmlRepackGemvMxfp4Q8Op monolith op; the
// SHARED body-leaf emitRepackCodebookGemvBodyMxfp4 (above) is now reached ONLY through the
// typed-region front door (emitTypedRepackGemvLoopBody's codebook branch, decode_model
// "mxfp4"). Byte-exact to the retired direct emitter was proven by intrinsic-sequence 0-diff.

mlir::LogicalResult VariantToEmitCFunc::emitRepackCodebookGemmBodyMxfp4(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQuantOffset,
    int64_t activationQuantOffset, llvm::ArrayRef<int8_t> codebook,
    int64_t weightInterleave, int64_t activationInterleave,
    int64_t half, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::StringRef l8 = coreLmul;
    llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";
    llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";
    llvm::StringRef maskN = l32 == "m4" ? "8" : "16";
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type f32m2Type =
        emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    mlir::Type u32m2Type =
        emitc::OpaqueType::get(ctx, ("vuint32" + l32 + "_t").str());
    mlir::Type boolMaskType =
        emitc::OpaqueType::get(ctx, ("vbool" + maskN + "_t").str());
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

    // block_mxfp4x16 repack facts (I4 mirror) are PARAMETERS now (the loop body op's
    // pinned attrs + the codebook core brick's kvalues, read by the codebook branch of
    // emitTypedRepackGemmLoopBody and passed in): qk, weightStride, activationStride,
    // weightQuantOffset, activationQuantOffset, codebook, weightInterleave,
    // activationInterleave, half.
    int64_t numHalves = weightInterleave / half;
    int64_t nibbleBytes = qk / 2;                    // 16
    int64_t activationHighRow = nibbleBytes;         // 16
    int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-mxfp4 output not pointer");

    mlir::Value vl8 = sizeLit(half);

    llvm::StringLiteral tableName = "weft_mxfp4_repack_kvalues";
    {
      std::string decl =
          ("static const int8_t " + tableName + "[16] = {").str();
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

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
    auto loadNibbles = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
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
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto codebookGather = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value off = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("codebook_index_offset"));
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value tablePtr =
                rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName);
            return {tablePtr, off, vl8};
          },
          llvm::StringRef("codebook_gather"));
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
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };

    // The E8M0 shared-exponent weight-scale strip (SHARED with the mxfp4 GEVM):
    // loads `half` E8M0 bytes, reconstructs 2^(e-128) per lane as an f32 vector.
    std::string vzextVf4Callee = ("__riscv_vzext_vf4_u32" + l32).str();
    std::string vandU32Callee = ("__riscv_vand_vx_u32" + l32).str();
    std::string vmvU32Callee = ("__riscv_vmv_v_x_u32" + l32).str();
    std::string vsllVvU32Callee = ("__riscv_vsll_vv_u32" + l32).str();
    std::string vsubU32Callee = ("__riscv_vsub_vx_u32" + l32).str();
    std::string vsllVxU32Callee = ("__riscv_vsll_vx_u32" + l32).str();
    std::string vmsltuCallee =
        ("__riscv_vmsltu_vx_u32" + l32 + "_b" + maskN).str();
    std::string vmergeU32Callee = ("__riscv_vmerge_vvm_u32" + l32).str();
    std::string vreinterpretCallee =
        ("__riscv_vreinterpret_v_u32" + l32 + "_f32" + l32).str();
    auto u32Imm = [&](llvm::StringRef callee, mlir::Type resTy, mlir::Value v,
                      llvm::StringRef imm) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, resTy, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value immV =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, imm)
                    .getResult();
            return {v, immV, vl8};
          });
    };
    auto e8m0ScaleStrip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_e8m0_scale_addr");
      mlir::Value eFull = bl;
      if (laneOff != 0)
        eFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(laneOff));
      mlir::Value eCast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, eFull).getResult();
      mlir::Value eU8 = emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                                       mlir::ValueRange{eCast, vl8}, opName, role,
                                       llvm::StringRef("e8m0_exponent_load"));
      mlir::Value e32 =
          emitOpaqueCall(rewriter, loc, u32m2Type, vzextVf4Callee,
                         mlir::ValueRange{eU8, vl8}, opName, role,
                         llvm::StringRef("e8m0_widen_u32"));
      step("e8m0_to_fp32_half_bits");
      mlir::Value emask = u32Imm(vandU32Callee, u32m2Type, e32, "0x1F");
      mlir::Value base = emitOpaqueCallBuilt(
          rewriter, loc, u32m2Type, vmvU32Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value lit = rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                                "0x00200000")
                                  .getResult();
            return {lit, vl8};
          });
      mlir::Value denorm =
          emitOpaqueCall(rewriter, loc, u32m2Type, vsllVvU32Callee,
                         mlir::ValueRange{base, emask, vl8}, opName, role);
      mlir::Value em1 = u32Imm(vsubU32Callee, u32m2Type, e32, "1");
      mlir::Value norm = u32Imm(vsllVxU32Callee, u32m2Type, em1, "23");
      mlir::Value mask = u32Imm(vmsltuCallee, boolMaskType, e32, "2");
      mlir::Value bits =
          emitOpaqueCall(rewriter, loc, u32m2Type, vmergeU32Callee,
                         mlir::ValueRange{norm, denorm, mask, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, f32m2Type, vreinterpretCallee,
                            mlir::ValueRange{bits}, opName, role,
                            llvm::StringRef("e8m0_reinterpret_f32"));
    };
    mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
    llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
    std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
    std::string mvI32Callee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");

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

            // Per-column activation scale d_y_c (4 fp16 at 0,2,4,6), cast to float.
            llvm::SmallVector<mlir::Value> aDf(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c) {
              mlir::Value aDc = emitOpaqueCallBuilt(
                  rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value aDPtr = al;
                    if (c != 0)
                      aDPtr = rewriter.create<emitc::AddOp>(
                          loc, activationPtrType, al, sizeLit(c * 2));
                    mlir::Value aDCast =
                        rewriter.create<emitc::CastOp>(loc, f16PtrType, aDPtr)
                            .getResult();
                    return {aDCast};
                  },
                  llvm::StringRef("act_scale_scalar"));
              aDf[c] =
                  rewriter.create<emitc::CastOp>(loc, floatType, aDc).getResult();
            }

            // SHARED E8M0 weight-scale strip (per-column-lane f32), reconstructed
            // ONCE per block and REUSED across the 4 activation columns.
            llvm::SmallVector<mlir::Value> bD(numHalves);
            for (int64_t h = 0; h < numHalves; ++h)
              bD[h] = e8m0ScaleStrip(bl, h * half);

            // Per-column i32 accumulator (single, no min).
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

            // ===== Nibble-step loop: SHARED codebook decode, per-column dot. ===
            for (int64_t i = 0; i < nibbleBytes; ++i) {
              llvm::SmallVector<mlir::Value> wLo(numHalves), wHi(numHalves);
              for (int64_t h = 0; h < numHalves; ++h) {
                step("weight_nibble_addr");
                mlir::Value packed =
                    loadNibbles(bl, weightQuantOffset + i * 16 + h * half);
                wLo[h] = codebookGather(u8Imm(vandCallee, packed, "0x0F"));
                wHi[h] = codebookGather(u8Imm(vsrlCallee, packed, "0x04"));
              }
              for (int64_t c = cLo; c < cHi; ++c) {
                step("act_quant_addr");
                mlir::Value aLo = i8Read(
                    al, activationQuantOffset + i * activationInterleave + c);
                mlir::Value aHi =
                    i8Read(al, activationQuantOffset +
                                   (i + activationHighRow) * activationInterleave +
                                   c);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value cur =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     sumiVar[c][h])
                          .getResult();
                  mlir::Value afterLo = vwaddw(cur, vwmul(aLo, wLo[h]));
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c][h], vwaddw(afterLo, vwmul(aHi, wHi[h])));
                }
              }
            }

            // ===== End-of-block per-column single-scale fold. =====
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value dC =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                                   mlir::ValueRange{bD[h], aDf[c], vl8}, opName,
                                   role);
                mlir::Value sumiV =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                        .getResult();
                mlir::Value sumiF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumiV, vl8}, opName, role);
                mlir::Value curF =
                    rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                        .getResult();
                mlir::Value nextF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                                   mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                                   role);
                rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], nextF);
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

    // The typed_repack_gemm_loop_body region op is RESULT-LESS (the per-column lane-wise
    // vse32 is the sink), so unlike the retired monolith direct emitter there is NO
    // trailing unused-result token to seed.
    return mlir::success();
  }

// NOTE (G3 retirement_batch 4): the mxfp4 repack GEMM direct emitter emitRepackGemmMxfp4Q8
// (+ its transitional thin shim) is RETIRED with the GgmlRepackGemmMxfp4Q8Op monolith op; the
// SHARED body-leaf emitRepackCodebookGemmBodyMxfp4 (above) is now reached ONLY through the
// typed-region front door (emitTypedRepackGemmLoopBody's codebook branch, decode_model
// "mxfp4"). Byte-exact to the retired direct emitter was proven by intrinsic-sequence 0-diff.

mlir::LogicalResult VariantToEmitCFunc::emitRepackCodebookGemvBodyIq4Xs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t weightQuantOffset, int64_t scalesLowOffset, int64_t scalesHighOffset,
    int64_t activationQuantOffset, int64_t nSubblocks,
    llvm::ArrayRef<int8_t> codebook, int64_t weightInterleave,
    int64_t half) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*): "mf2" (default
    // RVV1.0 fractional chain) or "m1" (RVV0.7 whole-LMUL chain). The block_iq4_xsx16
    // repack facts + the signed-6 scale offsets + the 16-entry non-linear int8 codebook
    // are PARAMETERS now (the loop body op's pinned attrs + the codebook core brick's
    // kvalues, read by the codebook super-block branch of emitTypedRepackGemvLoopBody).
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

    // block_iq4_xsx16 repack facts are PARAMETERS now (the loop body op's pinned attrs +
    // the codebook core brick's kvalues, passed in by the codebook super-block branch of
    // emitTypedRepackGemvLoopBody): qk, weightStride, activationStride, weightQuantOffset,
    // scalesLowOffset (scales_l LOW pair region), scalesHighOffset (scales_h HIGH 2-bit
    // region), activationQuantOffset, nSubblocks, codebook, weightInterleave, half.
    int64_t numHalves = weightInterleave / half;
    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t nibblesPerSub = subBlockSize / 2;        // 16

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(
          loc, "repack-gemv-iq4_xs output not pointer");

    mlir::Value vl8 = sizeLit(half);

    llvm::StringLiteral tableName = "weft_iq4_xs_repack_kvalues";
    {
      std::string decl =
          ("static const int8_t " + tableName + "[16] = {").str();
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

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
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    std::string vsllCallee = ("__riscv_vsll_vx_u8" + l8).str();
    std::string vorCallee = ("__riscv_vor_vv_u8" + l8).str();
    std::string vsubI8Callee = ("__riscv_vsub_vx_i8" + l8).str();
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
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    // Unpack the SIGNED 6-bit per-sub-block scale strip: low 4 bits from the
    // scales_l pair byte, high 2 bits from the sh_lo/sh_hi byte, OR them, then
    // vsub 32 (the -32 K-quant bias) -> a signed i8 scale in [-32,31], widened
    // to i32 for the vmacc scale-weight. NO min term (iq4_xs is scale-only).
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto unpackSignedScale = [&](mlir::Value bl, int64_t ib,
                                 int64_t h) -> mlir::Value {
      int64_t pair = ib / 2;
      mlir::Value loByte =
          loadU8Strip(bl, scalesLowOffset + pair * 16 + h * half);
      mlir::Value low4 = (ib % 2 == 0) ? u8Imm(vandCallee, loByte, "0x0F")
                                       : u8Imm(vsrlCallee, loByte, "4");
      int64_t shByte = scalesHighOffset + (ib < 4 ? 0 : 16) + h * half;
      mlir::Value hiByte = loadU8Strip(bl, shByte);
      mlir::Value high2 = u8Imm(
          vandCallee, u8Imm(vsrlCallee, hiByte, std::to_string(2 * (ib % 4))),
          "0x03");
      mlir::Value ls = u8Or(u8Imm(vsllCallee, high2, "4"), low4);
      mlir::Value lsI8 = emitOpaqueCall(rewriter, loc, i8mf2Type,
                                        reinterpretCallee,
                                        mlir::ValueRange{ls}, opName, role);
      mlir::Value signed8 = emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vsubI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value bias =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "32")
                    .getResult();
            return {lsI8, bias, vl8};
          });
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{signed8, vl8}, opName, role,
                            llvm::StringRef("signed_scale_widen"));
    };
    // The REAL codebook GATHER (memory vluxei16), identical to iq4_nl.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto codebookGather = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value off = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("codebook_index_offset"));
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value tablePtr =
                rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName);
            return {tablePtr, off, vl8};
          },
          llvm::StringRef("codebook_gather"));
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
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };
    // vmacc_vv i32: block_sumi += scale_i32 * sub_i32 (the SIGNED per-sub-block
    // scale weights the i32 sub-block dot into the block accumulator).
    std::string vmaccVVCallee = ("__riscv_vmacc_vv_i32" + l32).str();
    auto vmaccVV = [&](mlir::Value acc, mlir::Value a,
                       mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmaccVVCallee,
                            mlir::ValueRange{acc, a, b, vl8}, opName, role);
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
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      if (laneOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(laneOff * 2));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
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

        // Per-strip i32 block accumulator (scale-weighted sub-block dots).
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Per-sub-block loop ib in 0..7. =====
        for (int64_t ib = 0; ib < nSubblocks; ++ib) {
          // Signed 6-bit scale strip per column (widened to i32).
          step("subblock_signed_scale");
          llvm::SmallVector<mlir::Value> scale32(numHalves);
          for (int64_t h = 0; h < numHalves; ++h)
            scale32[h] = unpackSignedScale(bl, ib, h);
          // i32 sub-block dot accumulator per strip (codebook products overflow
          // i16, so the 32-element dot is accumulated in i32 directly).
          llvm::SmallVector<mlir::Value> subVar;
          for (int64_t h = 0; h < numHalves; ++h) {
            auto sv = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(i32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
            subVar.push_back(sv);
          }
          for (int64_t i = 0; i < nibblesPerSub; ++i) {
            for (int64_t h = 0; h < numHalves; ++h) {
              step("weight_nibble_addr");
              mlir::Value packed = loadU8Strip(
                  bl, weightQuantOffset + (ib * nibblesPerSub + i) * 16 +
                          h * half);
              mlir::Value wLo =
                  codebookGather(u8Imm(vandCallee, packed, "0x0F"));
              mlir::Value wHi =
                  codebookGather(u8Imm(vsrlCallee, packed, "0x04"));
              step("act_quant_addr");
              mlir::Value aLo =
                  i8Read(al, activationQuantOffset + ib * subBlockSize + i);
              mlir::Value aHi =
                  i8Read(al, activationQuantOffset + ib * subBlockSize + i +
                                 nibblesPerSub);
              mlir::Value cur =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[h])
                      .getResult();
              mlir::Value afterLo = vwaddw(cur, vwmul(aLo, wLo));
              rewriter.create<emitc::AssignOp>(loc, subVar[h],
                                               vwaddw(afterLo, vwmul(aHi, wHi)));
            }
          }
          step("subblock_scale_fold");
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value subV =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[h])
                    .getResult();
            mlir::Value curB =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(
                loc, sumiVar[h], vmaccVV(curB, scale32[h], subV));
          }
        }

        // ===== End-of-block fold: sumf += cvt(sumi) * (d_x * d_y). =====
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dStrip = loadF16Strip(bl, h * half);
          mlir::Value dW = emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                                          mlir::ValueRange{dStrip, vl8}, opName,
                                          role);
          mlir::Value d0 = emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                                          mlir::ValueRange{dW, aD, vl8}, opName,
                                          role);
          mlir::Value sumiV =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult();
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumiV, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccVVCallee,
                             mlir::ValueRange{curF, sumiF, d0, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], nextF);
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

    // The typed_repack_gemv_loop_body region op is RESULT-LESS (the per-strip lane-wise
    // vse32 is the sink), so unlike the retired monolith direct emitter there is NO
    // trailing unused-result token to seed.
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitRepackCodebookGemmBodyIq4Xs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t weightQuantOffset,
    int64_t scalesLowOffset, int64_t scalesHighOffset,
    int64_t activationQuantOffset, int64_t nSubblocks,
    llvm::ArrayRef<int8_t> codebook, int64_t weightInterleave,
    int64_t activationInterleave, int64_t half, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*): "mf2" (default
    // RVV1.0 fractional chain) or "m1" (RVV0.7 whole-LMUL chain). The block_iq4_xsx16
    // repack facts + the signed-6 scale offsets + the 16-entry non-linear int8 codebook
    // are PARAMETERS now (the loop body op's pinned attrs + the codebook core brick's
    // kvalues, read by the codebook super-block branch of emitTypedRepackGemmLoopBody).
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

    // block_iq4_xsx16 repack facts are PARAMETERS now (the loop body op's pinned attrs +
    // the codebook core brick's kvalues, passed in by the codebook super-block branch of
    // emitTypedRepackGemmLoopBody): qk, weightStride, activationStride, weightQuantOffset,
    // scalesLowOffset (scales_l LOW pair region), scalesHighOffset (scales_h HIGH 2-bit
    // region), activationQuantOffset, nSubblocks, codebook, weightInterleave,
    // activationInterleave, half.
    int64_t numHalves = weightInterleave / half;
    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t nibblesPerSub = subBlockSize / 2;        // 16
    int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(
          loc, "repack-gemm-iq4_xs output not pointer");

    mlir::Value vl8 = sizeLit(half);

    llvm::StringLiteral tableName = "weft_iq4_xs_repack_kvalues";
    {
      std::string decl =
          ("static const int8_t " + tableName + "[16] = {").str();
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

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
    std::string vsllCallee = ("__riscv_vsll_vx_u8" + l8).str();
    std::string vorCallee = ("__riscv_vor_vv_u8" + l8).str();
    std::string vsubI8Callee = ("__riscv_vsub_vx_i8" + l8).str();
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
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto unpackSignedScale = [&](mlir::Value bl, int64_t ib,
                                 int64_t h) -> mlir::Value {
      int64_t pair = ib / 2;
      mlir::Value loByte =
          loadU8Strip(bl, scalesLowOffset + pair * 16 + h * half);
      mlir::Value low4 = (ib % 2 == 0) ? u8Imm(vandCallee, loByte, "0x0F")
                                       : u8Imm(vsrlCallee, loByte, "4");
      int64_t shByte = scalesHighOffset + (ib < 4 ? 0 : 16) + h * half;
      mlir::Value hiByte = loadU8Strip(bl, shByte);
      mlir::Value high2 = u8Imm(
          vandCallee, u8Imm(vsrlCallee, hiByte, std::to_string(2 * (ib % 4))),
          "0x03");
      mlir::Value ls = u8Or(u8Imm(vsllCallee, high2, "4"), low4);
      mlir::Value lsI8 = emitOpaqueCall(rewriter, loc, i8mf2Type,
                                        reinterpretCallee,
                                        mlir::ValueRange{ls}, opName, role);
      mlir::Value signed8 = emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, vsubI8Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value bias =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "32")
                    .getResult();
            return {lsI8, bias, vl8};
          });
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{signed8, vl8}, opName, role,
                            llvm::StringRef("signed_scale_widen"));
    };
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto codebookGather = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value off = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("codebook_index_offset"));
      return emitOpaqueCallBuilt(
          rewriter, loc, i8mf2Type, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value tablePtr =
                rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName);
            return {tablePtr, off, vl8};
          },
          llvm::StringRef("codebook_gather"));
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
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };
    std::string vmaccVVCallee = ("__riscv_vmacc_vv_i32" + l32).str();
    auto vmaccVV = [&](mlir::Value acc, mlir::Value a,
                       mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmaccVVCallee,
                            mlir::ValueRange{acc, a, b, vl8}, opName, role);
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
    std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
    auto loadF16Strip = [&](mlir::Value bl, int64_t laneOff) -> mlir::Value {
      step("weight_scale_addr");
      mlir::Value dFull = bl;
      if (laneOff != 0)
        dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(laneOff * 2));
      mlir::Value dCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
      return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                            mlir::ValueRange{dCast, vl8}, opName, role);
    };
    std::string vfwcvtCallee = ("__riscv_vfwcvt_f_f_v_f32" + l32).str();
    std::string vfmulVfCallee = ("__riscv_vfmul_vf_f32" + l32).str();
    std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
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
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value dStrip = loadF16Strip(bl, h * half);
              dF32[h] = emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                                       mlir::ValueRange{dStrip, vl8}, opName,
                                       role);
            }

            // Per-column i32 block accumulator.
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

            // ===== Per-sub-block loop ib in 0..7. =====
            for (int64_t ib = 0; ib < nSubblocks; ++ib) {
              // SHARED signed scale strip per column-lane (reused over columns).
              step("subblock_signed_scale");
              llvm::SmallVector<mlir::Value> scale32(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                scale32[h] = unpackSignedScale(bl, ib, h);
              // Per-column i32 sub-block dot accumulator.
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> subVar(
                  activationInterleave);
              for (int64_t c = cLo; c < cHi; ++c)
                for (int64_t h = 0; h < numHalves; ++h) {
                  auto sv = rewriter.create<emitc::VariableOp>(
                      loc, emitc::LValueType::get(i32m2Type),
                      emitc::OpaqueAttr::get(ctx, ""));
                  rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
                  subVar[c].push_back(sv);
                }
              for (int64_t i = 0; i < nibblesPerSub; ++i) {
                // SHARED codebook decode per (ib,i,h), reused over columns.
                llvm::SmallVector<mlir::Value> wLo(numHalves), wHi(numHalves);
                for (int64_t h = 0; h < numHalves; ++h) {
                  step("weight_nibble_addr");
                  mlir::Value packed = loadU8Strip(
                      bl, weightQuantOffset + (ib * nibblesPerSub + i) * 16 +
                              h * half);
                  wLo[h] = codebookGather(u8Imm(vandCallee, packed, "0x0F"));
                  wHi[h] = codebookGather(u8Imm(vsrlCallee, packed, "0x04"));
                }
                for (int64_t c = cLo; c < cHi; ++c) {
                  step("act_quant_addr");
                  int64_t posLo = ib * subBlockSize + i;
                  int64_t posHi = ib * subBlockSize + i + nibblesPerSub;
                  mlir::Value aLo = i8Read(
                      al, activationQuantOffset + posLo * activationInterleave +
                              c);
                  mlir::Value aHi = i8Read(
                      al, activationQuantOffset + posHi * activationInterleave +
                              c);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value cur =
                        rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                       subVar[c][h])
                            .getResult();
                    mlir::Value afterLo = vwaddw(cur, vwmul(aLo, wLo[h]));
                    rewriter.create<emitc::AssignOp>(
                        loc, subVar[c][h], vwaddw(afterLo, vwmul(aHi, wHi[h])));
                  }
                }
              }
              step("subblock_scale_fold");
              for (int64_t c = cLo; c < cHi; ++c)
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value subV =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     subVar[c][h])
                          .getResult();
                  mlir::Value curB =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     sumiVar[c][h])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c][h], vmaccVV(curB, scale32[h], subV));
                }
            }

            // ===== End-of-block per-column fold: sumf += cvt(sumi)*(d_x*d_y). ==
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value d0 =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                                   mlir::ValueRange{dF32[h], aD[c], vl8}, opName,
                                   role);
                mlir::Value sumiV =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                        .getResult();
                mlir::Value sumiF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumiV, vl8}, opName, role);
                mlir::Value curF =
                    rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                        .getResult();
                mlir::Value nextF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccVVCallee,
                                   mlir::ValueRange{curF, sumiF, d0, vl8}, opName,
                                   role);
                rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], nextF);
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

    // The typed_repack_gemm_loop_body region op is RESULT-LESS (the per-column lane-wise
    // vse32 is the sink), so unlike the retired monolith direct emitter there is NO
    // trailing unused-result token to seed. The GEMM body ships PLAIN/UNTILED: iq4_xs
    // already sits at the <=32-vreg cliff, so S6 output tiling is a structural no-op.
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
