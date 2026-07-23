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

// Grid-family repack artifact consumers sharing the typed grid plan.

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvIq2DualScaleQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value columnCount,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t gridIdxOffset,
    int64_t lsOffset, int64_t signOffset, int64_t activationQuantOffset,
    int64_t nSubblocks, int64_t weightInterleave, int64_t half) const {
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
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t numGroups = subBlockSize / 8;            // 4 grid entries / sub-block
    int64_t groupsPerHalf = numGroups / 2;           // 2
    int64_t numHalves = weightInterleave / half;     // strip halves (VLEN split)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemv-iq2-dual output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The FIXED canonical GRID table + the SIGN plane, emitted ONCE (byte-exact
    // anchors the block-dot iq2_xs/iq2_s paths use). C4a: the DECL bodies (the
    // literals) stay in this Conversion layer, but WHICH tables and their NAMES are
    // now registry DATA -- no Iq2DualGridVariant hard-select.
    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "grid decode plan has no emit-period table decls");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, plan.gridArrayName);
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();
    mlir::Value signsI8Ptr =
        rewriter.create<emitc::LiteralOp>(loc, i8PtrType, plan.signArrayName);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load / gather helpers (byte-identical to iq2_xxs, plus the
    // u16 grid-index strip load the 9/10-bit index requires) ----
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
    // u16 grid-index strip: the 9/10-bit index cannot live in a byte, so it is a
    // uint16 lane loaded DIRECTLY (vle16 -- NO vzext).
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    std::string u16LoadCallee = riscvIntrinsicName("vle", 16, l16, "u16");
    auto loadU16Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u16m1Type, u16LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role,
                            llvm::StringRef("grid_index_u16_strip"));
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
    // Dual ls scale strip: ls[ib32][gh] int8 [1,31] widened to i32 (sext).
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t gh,
                         int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + (ib * 2 + gh) * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    // idx*8 byte-offset from a u16 index vector (grid: already u16, vsll only).
    auto shiftIdxU16 = [&](mlir::Value idxU16) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {idxU16, three, vl8};
          });
    };
    // sel*8 byte-offset from a u8 sign-selector vector (vzext then vsll).
    auto idxBaseFromU8 = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value w16 = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("sign_index_widen"));
      return shiftIdxU16(w16);
    };
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto gatherByte = [&](mlir::Value tablePtr, mlir::Value base,
                          int64_t j) -> mlir::Value {
      mlir::Value off = base;
      if (j != 0)
        off = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value jv =
                  rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                    std::to_string(j))
                      .getResult();
              return {base, jv, vl8};
            });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, gatherCallee,
                            mlir::ValueRange{tablePtr, off, vl8}, opName, role,
                            llvm::StringRef("grid_sign_gather"));
    };
    std::string vmulI8Callee = ("__riscv_vmul_vv_i8" + l8).str();
    auto signFold = [&](mlir::Value grid, mlir::Value signs) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, vmulI8Callee,
                            mlir::ValueRange{grid, signs, vl8}, opName, role,
                            llvm::StringRef("sign_fold_onto_grid"));
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

        // Per-strip i32 block accumulator (dual-ls-weighted sub-block dots).
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
        }

        // ===== Per-sub-block loop ib in 0..7, each split into TWO ls halves. =====
        for (int64_t ib = 0; ib < nSubblocks; ++ib) {
          for (int64_t gh = 0; gh < 2; ++gh) {
            step("subblock_ls_scale");
            llvm::SmallVector<mlir::Value> ls32(numHalves);
            for (int64_t h = 0; h < numHalves; ++h)
              ls32[h] = lsScale32(bl, ib, gh, h);
            // i32 group-half dot accumulator per strip.
            llvm::SmallVector<mlir::Value> subVar;
            for (int64_t h = 0; h < numHalves; ++h) {
              auto sv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
              subVar.push_back(sv);
            }
            // ===== The 2 grid GROUPS of this half. =====
            for (int64_t gi = 0; gi < groupsPerHalf; ++gi) {
              int64_t grp = gh * groupsPerHalf + gi;
              step("grid_sign_group");
              llvm::SmallVector<mlir::Value> gridBase(numHalves),
                  signBase(numHalves);
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value gidxU16 = loadU16Strip(
                    bl, gridIdxOffset +
                            ((ib * numGroups + grp) * 16 + h * half) * 2);
                mlir::Value sselU8 = loadU8Strip(
                    bl, signOffset + (ib * numGroups + grp) * 16 + h * half);
                gridBase[h] = shiftIdxU16(gidxU16);
                signBase[h] = idxBaseFromU8(sselU8);
              }
              for (int64_t j = 0; j < 8; ++j) {
                step("act_quant_addr");
                int64_t k = ib * subBlockSize + grp * 8 + j;
                mlir::Value aq = i8Read(al, activationQuantOffset + k);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value gridV = gatherByte(gridI8Ptr, gridBase[h], j);
                  mlir::Value signV = gatherByte(signsI8Ptr, signBase[h], j);
                  mlir::Value w = signFold(gridV, signV);
                  mlir::Value cur =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[h])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(loc, subVar[h],
                                                   vwaddw(cur, vwmul(aq, w)));
                }
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
              rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                               vmaccVV(curB, ls32[h], subV));
            }
          }
        }

        // ===== End-of-block fold: sumf += cvt(sumi) * (d_x * d_y). NO min. =====
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

      // Per-strip store: out[x*16 + h*half] = 0.125f * sumf_h.  (the iq2 1/8.)
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
        mlir::Value scaled = emitOpaqueCallBuilt(
            rewriter, loc, f32m2Type, vfmulVfCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value eighth =
                  rewriter.create<emitc::LiteralOp>(loc, floatType,
                                                    plan.storeScaleLiteral)
                      .getResult();
              return {sumfVal, eighth, vl8};
            },
            weft::gridStoreScaleRole(plan.storeScaleLiteral));
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, scaled, vl8}, opName, role);
      }
    }

    // RESULT-LESS (no monolith token): the front-door typed_repack_gemv_loop_body region
    // has NO result -- the lane-wise vector store is the sole sink (byte-exact to the
    // retired direct emitter modulo the dropped dead result-token vmv).
    return mlir::success();
  }

// The iq2_xs/iq2_s x q8_K 16x1-REPACKED PREFILL GEMM shared body. The prefill
// sibling of emitRepackGemvIq2DualScaleQ8K: the SAME dual-ls grid GATHER + sign
// GATHER + 0.125 fold, with the grid+sign weight decode AMORTIZED across the 4
// interleaved block_q8_Kx4 activation columns (4 fp32 d at +0, int8 quants at +16
// as pos*4+c).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmIq2DualScaleQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value rowCount,
    mlir::Value columnCount, mlir::Value outputRowStride, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef coreLmul, int64_t qk, int64_t weightStride,
    int64_t activationStride, int64_t gridIdxOffset, int64_t lsOffset,
    int64_t signOffset, int64_t activationQuantOffset, int64_t nSubblocks,
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
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t numGroups = subBlockSize / 8;            // 4
    int64_t groupsPerHalf = numGroups / 2;           // 2
    int64_t numHalves = weightInterleave / half;
    int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-iq2-dual output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // C4a: the GRID + SIGN tables are named by the CLOSED registry plan (data), not
    // hard-selected by an Iq2DualGridVariant enum. The DECL bodies stay here.
    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "grid decode plan has no emit-period table decls");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, plan.gridArrayName);
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();
    mlir::Value signsI8Ptr =
        rewriter.create<emitc::LiteralOp>(loc, i8PtrType, plan.signArrayName);

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
    std::string u16LoadCallee = riscvIntrinsicName("vle", 16, l16, "u16");
    auto loadU16Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u16m1Type, u16LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role,
                            llvm::StringRef("grid_index_u16_strip"));
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
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t gh,
                         int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + (ib * 2 + gh) * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    auto shiftIdxU16 = [&](mlir::Value idxU16) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {idxU16, three, vl8};
          });
    };
    auto idxBaseFromU8 = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value w16 = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("sign_index_widen"));
      return shiftIdxU16(w16);
    };
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto gatherByte = [&](mlir::Value tablePtr, mlir::Value base,
                          int64_t j) -> mlir::Value {
      mlir::Value off = base;
      if (j != 0)
        off = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value jv =
                  rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                    std::to_string(j))
                      .getResult();
              return {base, jv, vl8};
            });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, gatherCallee,
                            mlir::ValueRange{tablePtr, off, vl8}, opName, role,
                            llvm::StringRef("grid_sign_gather"));
    };
    std::string vmulI8Callee = ("__riscv_vmul_vv_i8" + l8).str();
    auto signFold = [&](mlir::Value grid, mlir::Value signs) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, vmulI8Callee,
                            mlir::ValueRange{grid, signs, vl8}, opName, role,
                            llvm::StringRef("sign_fold_onto_grid"));
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

            // Per-column i32 block accumulator (dual-ls-weighted sub-block dots).
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

            // ===== Per-sub-block loop ib, each split into TWO ls halves. =====
            for (int64_t ib = 0; ib < nSubblocks; ++ib) {
              for (int64_t gh = 0; gh < 2; ++gh) {
                step("subblock_ls_scale");
                llvm::SmallVector<mlir::Value> ls32(numHalves);
                for (int64_t h = 0; h < numHalves; ++h)
                  ls32[h] = lsScale32(bl, ib, gh, h);
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
                // ===== The 2 grid GROUPS of this half. SHARED weight decode. ==
                for (int64_t gi = 0; gi < groupsPerHalf; ++gi) {
                  int64_t grp = gh * groupsPerHalf + gi;
                  step("grid_sign_group");
                  llvm::SmallVector<mlir::Value> gridBase(numHalves),
                      signBase(numHalves);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value gidxU16 = loadU16Strip(
                        bl, gridIdxOffset +
                                ((ib * numGroups + grp) * 16 + h * half) * 2);
                    mlir::Value sselU8 = loadU8Strip(
                        bl, signOffset + (ib * numGroups + grp) * 16 + h * half);
                    gridBase[h] = shiftIdxU16(gidxU16);
                    signBase[h] = idxBaseFromU8(sselU8);
                  }
                  for (int64_t j = 0; j < 8; ++j) {
                    int64_t k = ib * subBlockSize + grp * 8 + j;
                    llvm::SmallVector<mlir::Value> w(numHalves);
                    for (int64_t h = 0; h < numHalves; ++h) {
                      mlir::Value gridV = gatherByte(gridI8Ptr, gridBase[h], j);
                      mlir::Value signV = gatherByte(signsI8Ptr, signBase[h], j);
                      w[h] = signFold(gridV, signV);
                    }
                    for (int64_t c = cLo; c < cHi; ++c) {
                      step("act_quant_addr");
                      mlir::Value aq = i8Read(
                          al,
                          activationQuantOffset + k * activationInterleave + c);
                      for (int64_t h = 0; h < numHalves; ++h) {
                        mlir::Value cur =
                            rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                           subVar[c][h])
                                .getResult();
                        rewriter.create<emitc::AssignOp>(
                            loc, subVar[c][h], vwaddw(cur, vwmul(aq, w[h])));
                      }
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
                        loc, sumiVar[c][h], vmaccVV(curB, ls32[h], subV));
                  }
              }
            }

            // ===== End-of-block per-column fold: sumf += cvt(sumi)*(d_x*d_y). ==
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value dStrip = loadF16Strip(bl, h * half);
                mlir::Value dW =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                                   mlir::ValueRange{dStrip, vl8}, opName, role);
                mlir::Value d0 = emitOpaqueCall(
                    rewriter, loc, f32m2Type, vfmulVfCallee,
                    mlir::ValueRange{dW, aD[c], vl8}, opName, role);
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

          // Per-column per-strip store: s + (y*4+c)*bs + x*16 + h*half, x 0.125.
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
              mlir::Value scaled = emitOpaqueCallBuilt(
                  rewriter, loc, f32m2Type, vfmulVfCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value eighth =
                        rewriter.create<emitc::LiteralOp>(loc, floatType,
                                                          plan.storeScaleLiteral)
                            .getResult();
                    return {sumfVal, eighth, vl8};
                  },
                  weft::gridStoreScaleRole(plan.storeScaleLiteral));
              emitOpaqueCallVoid(rewriter, loc, vseCallee,
                                 mlir::ValueRange{dst, scaled, vl8}, opName,
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

    // RESULT-LESS (no monolith token): the front-door typed_repack_gemm_loop_body region
    // has NO result -- the lane-wise vector store is the sole sink (byte-exact to the
    // retired direct emitter modulo the dropped dead result-token vmv).
    return mlir::success();
  }

// NOTE (G3 M4 iq2-grid front-door, cells iq2_xs + iq2_s): the four thin direct-emit
// dispatch entry points emitRepackGem{v,m}Iq2{Xs,S}Q8K + their monolith ops
// (weft_rvv.repack_gem{v,m}_iq2_{xs,s}_q8_K) + recognizers (isRepackGem{v,m}Iq2{Xs,S}Q8KBody)
// are RETIRED: the iq2_xs / iq2_s repack GEVM/GEMM now flow through the typed-region front
// door (grid branch of emitTypedRepackGem{v,m}LoopBody -> emitRepackGem{v,m}Iq2DualScaleQ8K,
// the SAME dual-ls body leaf now result-less + fact-parameterized). The iq2 grid family is
// COMPLETE at the front door.

// ---------------------------------------------------------------------------
// C4a-2: the iq1_s TERNARY-DELTA grid repack leaves (GEVM + GEMM).
//
// These implement ggml_vec_dot_iq1_s_q8_K's arithmetic on the REPACKED
// block_iq1_sx16 strip. The reference scalar shape (ggml) is:
//
//   for each super-block i:
//     sumi = 0; sumi1 = 0;
//     for ib in 0..7:
//       ls    = 2*((qh[ib] >> 12) & 7) + 1;
//       delta = qh[ib] & 0x8000 ? -1 : 1;
//       lsum  = sum_{l=0..3} sum_{j=0..7} q8[..] * grid(qs[l] | ((qh>>3l&7)<<8))[j];
//       sumi  += ls * lsum;
//       sumi1 += ls * delta * (bsums[2*ib] + bsums[2*ib+1]);
//     sumf += d_x*d_y * (sumi + IQ1S_DELTA * sumi1);      // IQ1S_DELTA = 0.125f
//
// The REPACK moves the qh word decode OFF the hot path: ls, delta and the 11-bit
// grid index are DERIVED once at repack time into three flat per-column strips, so
// the kernel only loads strips and gathers. What stays on the hot path is exactly
// the arithmetic above, per lane.
//
// Byte-exactness argument (integer domain): the ls-weighted grid dot is accumulated
// per sub-block into a private i32 `sub` and folded with ONE vmacc, so the addend
// SET and the ls multiply are the reference's; integer add is associative, so lane
// order is free. The delta term is a pure integer product summed per sub-block.
// The oracle (tools/oracle-repack/oracle_repack_iq1_s.cpp) checks BOTH integer
// accumulators (sumi, sumi1) byte-exact against a reference that decodes the
// ORIGINAL packed block_iq1_s -- a different layout and a different qh decode --
// so agreement is real evidence, not a shared-implementation tautology.
// ---------------------------------------------------------------------------

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvIq1SQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value columnCount,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t gridIdxOffset,
    int64_t lsOffset, int64_t deltaOffset, int64_t activationQuantOffset,
    int64_t activationBsumsOffset, int64_t nSubblocks, int64_t weightInterleave,
    int64_t half) const {
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
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int16_t"));
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t subBlockSize = qk / nSubblocks;      // 32
    int64_t numGroups = subBlockSize / 8;        // 4 grid entries / sub-block
    int64_t numHalves = weightInterleave / half; // strip halves (VLEN split)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemv-iq1-s output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The FIXED canonical 2048-entry TERNARY grid, emitted ONCE. A TernaryDelta
    // plan names NO sign table -- there is none: the grid bytes ARE the signed
    // weights, so no signs pointer is materialized here at all.
    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "grid decode plan has no emit-period table decls");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, plan.gridArrayName);
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // u16 grid-index strip: the 11-bit index cannot live in a byte, so it is a
    // uint16 lane loaded DIRECTLY (vle16 -- NO vzext), exactly like iq2_xs/iq2_s.
    std::string u16LoadCallee = riscvIntrinsicName("vle", 16, l16, "u16");
    auto loadU16Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u16m1Type, u16LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role,
                            llvm::StringRef("grid_index_u16_strip"));
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
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    // SINGLE ls per sub-block: ls[ib][lane] int8 in [1,15] widened to i32 (sext).
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    // The per-sub-block +-1 DELTA strip (the TernaryDelta sign-plane SLOT), i32.
    auto deltaVal32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value dI8 = loadI8Strip(bl, deltaOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{dI8, vl8}, opName, role,
                            llvm::StringRef("subblock_delta_widen"));
    };
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    // idx*8 byte-offset from a u16 index vector. Max idx 2047 -> 2047*8 = 16376,
    // well inside u16 (the SAME bound the iq1_s block-dot path relies on).
    auto shiftIdxU16 = [&](mlir::Value idxU16) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {idxU16, three, vl8};
          });
    };
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto gatherByte = [&](mlir::Value tablePtr, mlir::Value base,
                          int64_t j) -> mlir::Value {
      mlir::Value off = base;
      if (j != 0)
        off = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value jv =
                  rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                    std::to_string(j))
                      .getResult();
              return {base, jv, vl8};
            });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, gatherCallee,
                            mlir::ValueRange{tablePtr, off, vl8}, opName, role,
                            llvm::StringRef("ternary_grid_gather"));
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
    // A scalar i16 read of an activation bsum (the delta term's ONLY input).
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
    std::string vmulVVI32Callee = ("__riscv_vmul_vv_i32" + l32).str();
    auto vmulVV = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmulVVI32Callee,
                            mlir::ValueRange{a, b, vl8}, opName, role,
                            llvm::StringRef("ls_delta_product"));
    };
    std::string vmaccVXCallee = ("__riscv_vmacc_vx_i32" + l32).str();
    // sumi1 += (ls*delta) * bsumPair, with bsumPair the SCALAR activation term.
    auto vmaccVX = [&](mlir::Value acc, mlir::Value scalar,
                       mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmaccVXCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role, llvm::StringRef("delta_bsum_accumulate"));
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
    std::string vfaddVVCallee = ("__riscv_vfadd_vv_f32" + l32).str();
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

        // The TWO per-strip i32 block accumulators: sumi (ls-weighted ternary
        // grid dot) and sumi1 (the ls*delta*bsum-pair term). Kept SEPARATE across
        // all 8 sub-blocks, exactly like ggml's scalar sumi/sumi1.
        llvm::SmallVector<mlir::Value> sumiVar, sumi1Var;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumiVar.push_back(sv);
          auto s1v = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, s1v, seedI32());
          sumi1Var.push_back(s1v);
        }

        // ===== Per-sub-block loop ib in 0..7 (SINGLE ls, 4 grid groups). =====
        for (int64_t ib = 0; ib < nSubblocks; ++ib) {
          step("subblock_ls_scale");
          llvm::SmallVector<mlir::Value> ls32(numHalves);
          for (int64_t h = 0; h < numHalves; ++h)
            ls32[h] = lsScale32(bl, ib, h);
          // i32 sub-block dot accumulator per strip.
          llvm::SmallVector<mlir::Value> subVar;
          for (int64_t h = 0; h < numHalves; ++h) {
            auto sv = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(i32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
            subVar.push_back(sv);
          }
          // ===== The 4 ternary grid GROUPS of this sub-block. =====
          for (int64_t grp = 0; grp < numGroups; ++grp) {
            step("ternary_grid_group");
            llvm::SmallVector<mlir::Value> gridBase(numHalves);
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value gidxU16 = loadU16Strip(
                  bl, gridIdxOffset + ((ib * numGroups + grp) * 16 + h * half) * 2);
              gridBase[h] = shiftIdxU16(gidxU16);
            }
            for (int64_t j = 0; j < 8; ++j) {
              step("act_quant_addr");
              int64_t k = ib * subBlockSize + grp * 8 + j;
              mlir::Value aq = i8Read(al, activationQuantOffset + k);
              for (int64_t h = 0; h < numHalves; ++h) {
                // NO sign gather + NO vmul: the gathered ternary grid byte IS
                // the signed weight (that is what TernaryDelta means).
                mlir::Value w = gatherByte(gridI8Ptr, gridBase[h], j);
                mlir::Value cur =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[h])
                        .getResult();
                rewriter.create<emitc::AssignOp>(loc, subVar[h],
                                                 vwaddw(cur, vwmul(aq, w)));
              }
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
            rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                             vmaccVV(curB, ls32[h], subV));
          }
          // ===== The DELTA term: sumi1 += ls*delta * (bsums[2ib]+bsums[2ib+1]).
          // bsums are per-ACTIVATION (scalar); ls/delta are per-column (vector).
          step("delta_bsum_pair");
          mlir::Value bsum0 =
              i16Read(al, activationBsumsOffset + (2 * ib + 0) * 2);
          mlir::Value bsum1 =
              i16Read(al, activationBsumsOffset + (2 * ib + 1) * 2);
          mlir::Value bsumPair =
              rewriter.create<emitc::AddOp>(loc, i32Type, bsum0, bsum1)
                  .getResult();
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value delta32 = deltaVal32(bl, ib, h);
            mlir::Value lsDelta = vmulVV(ls32[h], delta32);
            mlir::Value cur1 =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumi1Var[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumi1Var[h],
                                             vmaccVX(cur1, bsumPair, lsDelta));
          }
        }

        // ===== End-of-block DeltaGrid fold (NOT the iq2 store-side 0.125):
        // sumf += (d_x*d_y) * (cvt(sumi) + 0.125f*cvt(sumi1)).
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
          mlir::Value sumi1V =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumi1Var[h])
                  .getResult();
          mlir::Value sumi1F =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumi1V, vl8}, opName, role);
          // 0.125f (IQ1S_DELTA) applied EXACTLY ONCE, to sumi1 only.
          step("iq1s_delta_scale");
          mlir::Value deltaScaled = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfmulVfCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value eighth =
                    rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f")
                        .getResult();
                return {sumi1F, eighth, vl8};
              },
              llvm::StringRef("eighth_scale"));
          mlir::Value inner =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfaddVVCallee,
                             mlir::ValueRange{sumiF, deltaScaled, vl8}, opName,
                             role, llvm::StringRef("delta_grid_inner"));
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccVVCallee,
                             mlir::ValueRange{curF, inner, d0, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], nextF);
        }
      }

      // Per-strip store: out[x*16 + h*half] = sumf_h. NO trailing 0.125 -- the
      // IQ1S_DELTA already rode the per-block sumi1 term above.
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

    // RESULT-LESS: the lane-wise vector store is the sole sink.
    return mlir::success();
  }

// The iq1_s x q8_K 16x1-REPACKED PREFILL GEMM body. The prefill sibling of
// emitRepackGemvIq1SQ8K: the SAME ternary grid GATHER + delta-bsum dual
// accumulator, with the weight decode AMORTIZED across the 4 interleaved
// block_q8_Kx4 activation columns (4 fp32 d at +0, int8 quants at +16 as pos*4+c,
// int16 bsums at +1040 as g16*4+c).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmIq1SQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value rowCount,
    mlir::Value columnCount, mlir::Value outputRowStride, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef coreLmul, int64_t qk, int64_t weightStride,
    int64_t activationStride, int64_t gridIdxOffset, int64_t lsOffset,
    int64_t deltaOffset, int64_t activationQuantOffset,
    int64_t activationBsumsOffset, int64_t nSubblocks, int64_t weightInterleave,
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
    mlir::Type i8mf2Type =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
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
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int16_t"));
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t subBlockSize = qk / nSubblocks;      // 32
    int64_t numGroups = subBlockSize / 8;        // 4
    int64_t numHalves = weightInterleave / half;
    int64_t columnsPerPass = coreLmul == "m1" ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-iq1-s output not pointer");

    mlir::Value vl8 = sizeLit(half);

    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "grid decode plan has no emit-period table decls");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, plan.gridArrayName);
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("row_group_count");
    mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, rowCount, sizeLit(activationInterleave));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    std::string u16LoadCallee = riscvIntrinsicName("vle", 16, l16, "u16");
    auto loadU16Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u16m1Type, u16LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role,
                            llvm::StringRef("grid_index_u16_strip"));
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
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    auto deltaVal32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value dI8 = loadI8Strip(bl, deltaOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{dI8, vl8}, opName, role,
                            llvm::StringRef("subblock_delta_widen"));
    };
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    auto shiftIdxU16 = [&](mlir::Value idxU16) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {idxU16, three, vl8};
          });
    };
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto gatherByte = [&](mlir::Value tablePtr, mlir::Value base,
                          int64_t j) -> mlir::Value {
      mlir::Value off = base;
      if (j != 0)
        off = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value jv =
                  rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                    std::to_string(j))
                      .getResult();
              return {base, jv, vl8};
            });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, gatherCallee,
                            mlir::ValueRange{tablePtr, off, vl8}, opName, role,
                            llvm::StringRef("ternary_grid_gather"));
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
    std::string vmulVVI32Callee = ("__riscv_vmul_vv_i32" + l32).str();
    auto vmulVV = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmulVVI32Callee,
                            mlir::ValueRange{a, b, vl8}, opName, role,
                            llvm::StringRef("ls_delta_product"));
    };
    std::string vmaccVXCallee = ("__riscv_vmacc_vx_i32" + l32).str();
    auto vmaccVX = [&](mlir::Value acc, mlir::Value scalar,
                       mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmaccVXCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role, llvm::StringRef("delta_bsum_accumulate"));
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
    std::string vfaddVVCallee = ("__riscv_vfadd_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // The loop-order schedule axis: identical to the iq2 GEMM sibling -- both
    // nestings emit a BYTE-IDENTICAL tile, only the two ForOp headers swap.
    auto emitAGroupBase = [&](mlir::Value y) -> mlir::Value {
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      return rewriter.create<emitc::AddOp>(loc, activationPtrType, activationBase,
                                           aGroupOff);
    };
    auto emitBGroupBase = [&](mlir::Value x) -> mlir::Value {
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      return rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase,
                                           bGroupOff);
    };

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

            // The TWO per-(column, strip) i32 block accumulators.
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumiVar(
                activationInterleave), sumi1Var(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                auto sv = rewriter.create<emitc::VariableOp>(
                    loc, emitc::LValueType::get(i32m2Type),
                    emitc::OpaqueAttr::get(ctx, ""));
                rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
                sumiVar[c].push_back(sv);
                auto s1v = rewriter.create<emitc::VariableOp>(
                    loc, emitc::LValueType::get(i32m2Type),
                    emitc::OpaqueAttr::get(ctx, ""));
                rewriter.create<emitc::AssignOp>(loc, s1v, seedI32());
                sumi1Var[c].push_back(s1v);
              }

            // ===== Per-sub-block loop ib (SINGLE ls, 4 groups, SHARED decode). =
            for (int64_t ib = 0; ib < nSubblocks; ++ib) {
              step("subblock_ls_scale");
              llvm::SmallVector<mlir::Value> ls32(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                ls32[h] = lsScale32(bl, ib, h);
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
              // ===== The 4 ternary grid GROUPS. The weight decode (gather) is
              // hoisted out of the column loop = the GEMM amortization. =====
              for (int64_t grp = 0; grp < numGroups; ++grp) {
                step("ternary_grid_group");
                llvm::SmallVector<mlir::Value> gridBase(numHalves);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value gidxU16 = loadU16Strip(
                      bl, gridIdxOffset +
                              ((ib * numGroups + grp) * 16 + h * half) * 2);
                  gridBase[h] = shiftIdxU16(gidxU16);
                }
                for (int64_t j = 0; j < 8; ++j) {
                  int64_t k = ib * subBlockSize + grp * 8 + j;
                  llvm::SmallVector<mlir::Value> w(numHalves);
                  for (int64_t h = 0; h < numHalves; ++h)
                    w[h] = gatherByte(gridI8Ptr, gridBase[h], j);
                  for (int64_t c = cLo; c < cHi; ++c) {
                    step("act_quant_addr");
                    mlir::Value aq = i8Read(
                        al,
                        activationQuantOffset + k * activationInterleave + c);
                    for (int64_t h = 0; h < numHalves; ++h) {
                      mlir::Value cur =
                          rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                         subVar[c][h])
                              .getResult();
                      rewriter.create<emitc::AssignOp>(
                          loc, subVar[c][h], vwaddw(cur, vwmul(aq, w[h])));
                    }
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
                      loc, sumiVar[c][h], vmaccVV(curB, ls32[h], subV));
                }
              // ===== The DELTA term per column. The ls*delta product is SHARED
              // across the 4 columns (a weight fact); only the bsum pair is
              // per-column. Interleaved q8_Kx4 bsums: bsums[g16*4 + c].
              step("delta_bsum_pair");
              llvm::SmallVector<mlir::Value> lsDelta(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                lsDelta[h] = vmulVV(ls32[h], deltaVal32(bl, ib, h));
              for (int64_t c = cLo; c < cHi; ++c) {
                mlir::Value bsum0 = i16Read(
                    al, activationBsumsOffset +
                            ((2 * ib + 0) * activationInterleave + c) * 2);
                mlir::Value bsum1 = i16Read(
                    al, activationBsumsOffset +
                            ((2 * ib + 1) * activationInterleave + c) * 2);
                mlir::Value bsumPair =
                    rewriter.create<emitc::AddOp>(loc, i32Type, bsum0, bsum1)
                        .getResult();
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value cur1 =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     sumi1Var[c][h])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(
                      loc, sumi1Var[c][h], vmaccVX(cur1, bsumPair, lsDelta[h]));
                }
              }
            }

            // ===== End-of-block per-column DeltaGrid fold. =====
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value dStrip = loadF16Strip(bl, h * half);
                mlir::Value dW =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                                   mlir::ValueRange{dStrip, vl8}, opName, role);
                mlir::Value d0 = emitOpaqueCall(
                    rewriter, loc, f32m2Type, vfmulVfCallee,
                    mlir::ValueRange{dW, aD[c], vl8}, opName, role);
                mlir::Value sumiV =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                        .getResult();
                mlir::Value sumiF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumiV, vl8}, opName, role);
                mlir::Value sumi1V =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                   sumi1Var[c][h])
                        .getResult();
                mlir::Value sumi1F =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumi1V, vl8}, opName, role);
                step("iq1s_delta_scale");
                mlir::Value deltaScaled = emitOpaqueCallBuilt(
                    rewriter, loc, f32m2Type, vfmulVfCallee, opName, role,
                    [&](mlir::OpBuilder &b,
                        mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                      mlir::Value eighth =
                          rewriter.create<emitc::LiteralOp>(loc, floatType,
                                                            "0.125f")
                              .getResult();
                      return {sumi1F, eighth, vl8};
                    },
                    llvm::StringRef("eighth_scale"));
                mlir::Value inner = emitOpaqueCall(
                    rewriter, loc, f32m2Type, vfaddVVCallee,
                    mlir::ValueRange{sumiF, deltaScaled, vl8}, opName, role,
                    llvm::StringRef("delta_grid_inner"));
                mlir::Value curF =
                    rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                        .getResult();
                mlir::Value nextF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccVVCallee,
                                   mlir::ValueRange{curF, inner, d0, vl8},
                                   opName, role);
                rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], nextF);
              }
          }

          // Per-column per-strip store: s + (y*4+c)*bs + x*16 + h*half. NO
          // trailing 0.125 (it rode the per-block sumi1 term).
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

    // RESULT-LESS: the lane-wise vector store is the sole sink.
    return mlir::success();
  }

// The iq1_m x q8_K 16x1-REPACKED DECODE GEVM body (C4a-3). iq1_s's sibling: the SAME
// already-signed ternary grid gather (no sign plane, no sign fold) and the SAME
// sumf += (d_x*d_y)*(cvt(accA) + 0.125f*cvt(accB)) block fold, with TWO structural
// changes that are why this is a leaf and not a parameter of emitRepackGemvIq1SQ8K:
//
//   * DUAL ls (plan.lsArity == Dual, REUSED from iq2_xs/iq2_s): ls1 folds groups 0-1,
//     ls2 folds groups 2-3, off the SAME (ib*2 + gh)*16 dual-ls strip shape.
//   * PER-GROUP delta over IN-KERNEL group-of-8 activation sums. The DELTA strip holds
//     FOUR +-1 per sub-block (INDEPENDENT qh bits 0x08/0x80 of qh[l/2]), and the delta
//     term's activation factor is sum_{j<8} q8[8*grp + j] -- accumulated here from the
//     SAME 8 activation scalars the grid dot already reads. It is NOT a bsums read, and
//     this leaf takes no bsums offset: block_q8_K's bsums are sums over groups of
//     SIXTEEN (ggml-common.h `int16_t bsums[QK_K/16]`) and the two 8-groups inside one
//     bsums entry carry INDEPENDENT delta signs, so a bsums entry cannot express it.
//
// ggml (ggml_vec_dot_iq1_m_q8_K) accumulates the delta term as
//     sum2[l/2] += lsum2(l)*delta[l];  sumi2 += sum2[0]*ls1 + sum2[1]*ls2
// and the grid term as
//     sum1[l/2] += lsum1(l);           sumi1 += sum1[0]*ls1 + sum1[1]*ls2.
// This leaf folds PER GROUP instead: sumi1 += ls[grp/2]*lsum1(grp) and
// sumi2 += (ls[grp/2]*delta[grp])*lsum2(grp). That is the SAME value, by integer
// distributivity -- ls[grp/2] is constant across the two groups ggml sums first, and
// every term here is an exact int32 (|lsum1| <= 8*127*1, |lsum2| <= 8*127, ls <= 15, 8
// sub-blocks x 4 groups => |sumi2| <= 8*4*15*1016 < 2^20), so no reassociation error
// and no overflow. The FLOAT fold is untouched, so byte-exactness is preserved.
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvIq1MQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value columnCount,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t gridIdxOffset,
    int64_t lsOffset, int64_t deltaOffset, int64_t activationQuantOffset,
    int64_t nSubblocks, int64_t weightInterleave, int64_t half) const {
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
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t subBlockSize = qk / nSubblocks;      // 32
    int64_t numGroups = subBlockSize / 8;        // 4 grid entries / sub-block
    int64_t numHalves = weightInterleave / half; // strip halves (VLEN split)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemv-iq1-m output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The FIXED canonical 2048-entry TERNARY grid, emitted ONCE. A TernaryDelta plan
    // names NO sign table -- there is none.
    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "grid decode plan has no emit-period table decls");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, plan.gridArrayName);
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // u16 grid-index strip: the 11-bit index cannot live in a byte, so it is a
    // uint16 lane loaded DIRECTLY (vle16 -- NO vzext), exactly like iq1_s/iq2_xs.
    std::string u16LoadCallee = riscvIntrinsicName("vle", 16, l16, "u16");
    auto loadU16Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u16m1Type, u16LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role,
                            llvm::StringRef("grid_index_u16_strip"));
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
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    // DUAL ls: gh selects ls1 (groups 0-1) / ls2 (groups 2-3) off the SAME
    // (ib*2 + gh)*16 strip shape the iq2 dual-ls leaf uses. int8 in [1,15] -> i32.
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t gh,
                         int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + (ib * 2 + gh) * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_dual_scale_widen"));
    };
    // The PER-GROUP +-1 DELTA strip (the TernaryDelta sign-plane SLOT), i32. FOUR per
    // sub-block -- this is the iq1_s delta axis at group granularity.
    auto deltaVal32 = [&](mlir::Value bl, int64_t ib, int64_t grp,
                          int64_t h) -> mlir::Value {
      mlir::Value dI8 =
          loadI8Strip(bl, deltaOffset + (ib * numGroups + grp) * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{dI8, vl8}, opName, role,
                            llvm::StringRef("group_delta_widen"));
    };
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    // idx*8 byte-offset from a u16 index vector. Max idx 2047 -> 2047*8 = 16376,
    // well inside u16 (the SAME I64x8 entry-width bound iq1_s relies on).
    auto shiftIdxU16 = [&](mlir::Value idxU16) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {idxU16, three, vl8};
          });
    };
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto gatherByte = [&](mlir::Value tablePtr, mlir::Value base,
                          int64_t j) -> mlir::Value {
      mlir::Value off = base;
      if (j != 0)
        off = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value jv =
                  rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                    std::to_string(j))
                      .getResult();
              return {base, jv, vl8};
            });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, gatherCallee,
                            mlir::ValueRange{tablePtr, off, vl8}, opName, role,
                            llvm::StringRef("ternary_grid_gather"));
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
    std::string vmulVVI32Callee = ("__riscv_vmul_vv_i32" + l32).str();
    auto vmulVV = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmulVVI32Callee,
                            mlir::ValueRange{a, b, vl8}, opName, role,
                            llvm::StringRef("ls_delta_product"));
    };
    std::string vmaccVXCallee = ("__riscv_vmacc_vx_i32" + l32).str();
    // sumi2 += (ls*delta) * groupSum, with groupSum the SCALAR in-kernel per-group-of-8
    // activation quant sum (NOT a bsums read -- see the leaf comment).
    auto vmaccVX = [&](mlir::Value acc, mlir::Value scalar,
                       mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmaccVXCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role, llvm::StringRef("delta_groupsum_accumulate"));
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
    // The ASSEMBLED fp16 d strip. block_iq1_m has NO inline d (its fp16 is four
    // nibbles scattered across the scales words); the repack assembled it, so the
    // kernel loads an ordinary inline fp16 strip like every other row.
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
    std::string vfaddVVCallee = ("__riscv_vfadd_vv_f32" + l32).str();
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

        // The TWO per-strip i32 block accumulators: sumi1 (the ls-weighted ternary
        // grid dot) and sumi2 (the ls*delta*groupSum term). Kept SEPARATE across all
        // 8 sub-blocks, exactly like ggml's scalar sumi1/sumi2.
        llvm::SmallVector<mlir::Value> sumi1Var, sumi2Var;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto sv = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
          sumi1Var.push_back(sv);
          auto s2v = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, s2v, seedI32());
          sumi2Var.push_back(s2v);
        }

        // ===== Per-sub-block loop ib in 0..7 (DUAL ls, 4 grid groups). =====
        for (int64_t ib = 0; ib < nSubblocks; ++ib) {
          step("subblock_dual_ls_scale");
          // ls32[gh][h]: gh=0 -> ls1 (groups 0-1), gh=1 -> ls2 (groups 2-3).
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> ls32(2);
          for (int64_t gh = 0; gh < 2; ++gh)
            for (int64_t h = 0; h < numHalves; ++h)
              ls32[gh].push_back(lsScale32(bl, ib, gh, h));
          // ===== The 4 ternary grid GROUPS of this sub-block. =====
          for (int64_t grp = 0; grp < numGroups; ++grp) {
            step("ternary_grid_group");
            // The ls half this group folds under: groups 0-1 -> ls1, 2-3 -> ls2.
            int64_t gh = grp / 2;
            llvm::SmallVector<mlir::Value> gridBase(numHalves);
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value gidxU16 = loadU16Strip(
                  bl, gridIdxOffset + ((ib * numGroups + grp) * 16 + h * half) * 2);
              gridBase[h] = shiftIdxU16(gidxU16);
            }
            // The per-group i32 grid-dot accumulator per strip (ggml's lsum1)...
            llvm::SmallVector<mlir::Value> subVar;
            for (int64_t h = 0; h < numHalves; ++h) {
              auto sv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
              subVar.push_back(sv);
            }
            // ...and the SCALAR per-group-of-8 activation sum (ggml's lsum2). This is
            // the delta term's activation factor, accumulated from the SAME 8 scalars
            // the grid dot reads -- the bsums plane cannot supply it (per-16).
            step("act_group8_sum");
            mlir::Value groupSum;
            for (int64_t j = 0; j < 8; ++j) {
              step("act_quant_addr");
              int64_t k = ib * subBlockSize + grp * 8 + j;
              mlir::Value aq = i8Read(al, activationQuantOffset + k);
              groupSum = j == 0 ? aq
                                : rewriter
                                      .create<emitc::AddOp>(loc, i32Type,
                                                            groupSum, aq)
                                      .getResult();
              for (int64_t h = 0; h < numHalves; ++h) {
                // NO sign gather + NO vmul: the gathered ternary grid byte IS the
                // signed weight (that is what TernaryDelta means).
                mlir::Value w = gatherByte(gridI8Ptr, gridBase[h], j);
                mlir::Value cur =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[h])
                        .getResult();
                rewriter.create<emitc::AssignOp>(loc, subVar[h],
                                                 vwaddw(cur, vwmul(aq, w)));
              }
            }
            // sumi1 += ls[gh] * lsum1(grp)  (ggml folds the two groups of an ls half
            // together first; identical by integer distributivity).
            step("group_scale_fold");
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value subV =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[h])
                      .getResult();
              mlir::Value curB =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumi1Var[h])
                      .getResult();
              rewriter.create<emitc::AssignOp>(loc, sumi1Var[h],
                                               vmaccVV(curB, ls32[gh][h], subV));
            }
            // sumi2 += (ls[gh]*delta[grp]) * lsum2(grp). groupSum is per-ACTIVATION
            // (scalar); ls/delta are per-column (vector).
            step("group_delta_fold");
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value delta32 = deltaVal32(bl, ib, grp, h);
              mlir::Value lsDelta = vmulVV(ls32[gh][h], delta32);
              mlir::Value cur2 =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumi2Var[h])
                      .getResult();
              rewriter.create<emitc::AssignOp>(loc, sumi2Var[h],
                                               vmaccVX(cur2, groupSum, lsDelta));
            }
          }
        }

        // ===== End-of-block DeltaGridGroupSum fold (the SAME expression as the
        // iq1_s DeltaGrid fold; only the way accB was BUILT differs):
        // sumf += (d_x*d_y) * (cvt(sumi1) + 0.125f*cvt(sumi2)).
        for (int64_t h = 0; h < numHalves; ++h) {
          mlir::Value dStrip = loadF16Strip(bl, h * half);
          mlir::Value dW = emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                                          mlir::ValueRange{dStrip, vl8}, opName,
                                          role);
          mlir::Value d0 = emitOpaqueCall(rewriter, loc, f32m2Type, vfmulVfCallee,
                                          mlir::ValueRange{dW, aD, vl8}, opName,
                                          role);
          mlir::Value sumi1V =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumi1Var[h])
                  .getResult();
          mlir::Value sumi1F =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumi1V, vl8}, opName, role);
          mlir::Value sumi2V =
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumi2Var[h])
                  .getResult();
          mlir::Value sumi2F =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumi2V, vl8}, opName, role);
          // 0.125f (IQ1M_DELTA) applied EXACTLY ONCE, to sumi2 only.
          step("iq1m_delta_scale");
          mlir::Value deltaScaled = emitOpaqueCallBuilt(
              rewriter, loc, f32m2Type, vfmulVfCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value eighth =
                    rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f")
                        .getResult();
                return {sumi2F, eighth, vl8};
              },
              llvm::StringRef("eighth_scale"));
          mlir::Value inner =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfaddVVCallee,
                             mlir::ValueRange{sumi1F, deltaScaled, vl8}, opName,
                             role, llvm::StringRef("delta_grid_inner"));
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[h])
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccVVCallee,
                             mlir::ValueRange{curF, inner, d0, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar[h], nextF);
        }
      }

      // Per-strip store: out[x*16 + h*half] = sumf_h. NO trailing 0.125 -- the
      // IQ1M_DELTA already rode the per-block sumi2 term above.
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

    // RESULT-LESS: the lane-wise vector store is the sole sink.
    return mlir::success();
  }

// The iq1_m x q8_K 16x1-REPACKED PREFILL GEMM body (C4a-3). The prefill sibling of
// emitRepackGemvIq1MQ8K: the SAME ternary grid GATHER + dual-ls + per-group-delta
// group-sum dual accumulator, with the weight decode AMORTIZED across the 4
// interleaved block_q8_Kx4 activation columns (4 fp32 d at +0, int8 quants at +16 as
// pos*4+c). Reads NO bsums -- the group-of-8 sums are accumulated per column from the
// interleaved quants (see the GEVM leaf for why bsums are inexpressive here), which is
// why the delta term's SCALAR factor is per-COLUMN while the ls*delta product stays
// SHARED across the 4 columns (a weight fact).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmIq1MQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value rowCount,
    mlir::Value columnCount, mlir::Value outputRowStride, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef coreLmul, int64_t qk, int64_t weightStride,
    int64_t activationStride, int64_t gridIdxOffset, int64_t lsOffset,
    int64_t deltaOffset, int64_t activationQuantOffset, int64_t nSubblocks,
    int64_t weightInterleave, int64_t activationInterleave, int64_t half,
    bool colGroupOuter) const {
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
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t subBlockSize = qk / nSubblocks;      // 32
    int64_t numGroups = subBlockSize / 8;        // 4
    int64_t numHalves = weightInterleave / half;
    int64_t columnsPerPass = coreLmul == "m1" ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-iq1-m output not pointer");

    mlir::Value vl8 = sizeLit(half);

    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "grid decode plan has no emit-period table decls");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, plan.gridArrayName);
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("row_group_count");
    mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, rowCount, sizeLit(activationInterleave));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    std::string u16LoadCallee = riscvIntrinsicName("vle", 16, l16, "u16");
    auto loadU16Strip = [&](mlir::Value base, int64_t byteOff) -> mlir::Value {
      mlir::Value full = base;
      if (byteOff != 0)
        full = rewriter.create<emitc::AddOp>(loc, weightPtrType, base,
                                             sizeLit(byteOff));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u16m1Type, u16LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role,
                            llvm::StringRef("grid_index_u16_strip"));
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
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t gh,
                         int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + (ib * 2 + gh) * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_dual_scale_widen"));
    };
    auto deltaVal32 = [&](mlir::Value bl, int64_t ib, int64_t grp,
                          int64_t h) -> mlir::Value {
      mlir::Value dI8 =
          loadI8Strip(bl, deltaOffset + (ib * numGroups + grp) * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{dI8, vl8}, opName, role,
                            llvm::StringRef("group_delta_widen"));
    };
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    auto shiftIdxU16 = [&](mlir::Value idxU16) -> mlir::Value {
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {idxU16, three, vl8};
          });
    };
    std::string gatherCallee =
        riscvIndexedMemoryIntrinsicName("vluxei", 16, "i8", l8);
    auto gatherByte = [&](mlir::Value tablePtr, mlir::Value base,
                          int64_t j) -> mlir::Value {
      mlir::Value off = base;
      if (j != 0)
        off = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value jv =
                  rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                    std::to_string(j))
                      .getResult();
              return {base, jv, vl8};
            });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, gatherCallee,
                            mlir::ValueRange{tablePtr, off, vl8}, opName, role,
                            llvm::StringRef("ternary_grid_gather"));
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
    std::string vmulVVI32Callee = ("__riscv_vmul_vv_i32" + l32).str();
    auto vmulVV = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmulVVI32Callee,
                            mlir::ValueRange{a, b, vl8}, opName, role,
                            llvm::StringRef("ls_delta_product"));
    };
    std::string vmaccVXCallee = ("__riscv_vmacc_vx_i32" + l32).str();
    auto vmaccVX = [&](mlir::Value acc, mlir::Value scalar,
                       mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vmaccVXCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role, llvm::StringRef("delta_groupsum_accumulate"));
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
    std::string vfaddVVCallee = ("__riscv_vfadd_vv_f32" + l32).str();
    llvm::StringRef floatReadCallee = "*(const float *)";
    mlir::Type floatPtrConstType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // The loop-order schedule axis: identical to the iq1_s / iq2 GEMM siblings --
    // both nestings emit a BYTE-IDENTICAL tile, only the two ForOp headers swap.
    auto emitAGroupBase = [&](mlir::Value y) -> mlir::Value {
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      return rewriter.create<emitc::AddOp>(loc, activationPtrType, activationBase,
                                           aGroupOff);
    };
    auto emitBGroupBase = [&](mlir::Value x) -> mlir::Value {
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      return rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase,
                                           bGroupOff);
    };

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

            // The TWO per-(column, strip) i32 block accumulators.
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumi1Var(
                activationInterleave), sumi2Var(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                auto sv = rewriter.create<emitc::VariableOp>(
                    loc, emitc::LValueType::get(i32m2Type),
                    emitc::OpaqueAttr::get(ctx, ""));
                rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
                sumi1Var[c].push_back(sv);
                auto s2v = rewriter.create<emitc::VariableOp>(
                    loc, emitc::LValueType::get(i32m2Type),
                    emitc::OpaqueAttr::get(ctx, ""));
                rewriter.create<emitc::AssignOp>(loc, s2v, seedI32());
                sumi2Var[c].push_back(s2v);
              }

            // ===== Per-sub-block loop ib (DUAL ls, 4 groups, SHARED decode). =====
            for (int64_t ib = 0; ib < nSubblocks; ++ib) {
              step("subblock_dual_ls_scale");
              llvm::SmallVector<llvm::SmallVector<mlir::Value>> ls32(2);
              for (int64_t gh = 0; gh < 2; ++gh)
                for (int64_t h = 0; h < numHalves; ++h)
                  ls32[gh].push_back(lsScale32(bl, ib, gh, h));
              // ===== The 4 ternary grid GROUPS. The weight decode (gather) is
              // hoisted out of the column loop = the GEMM amortization. =====
              for (int64_t grp = 0; grp < numGroups; ++grp) {
                step("ternary_grid_group");
                int64_t gh = grp / 2;
                llvm::SmallVector<mlir::Value> gridBase(numHalves);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value gidxU16 = loadU16Strip(
                      bl, gridIdxOffset +
                              ((ib * numGroups + grp) * 16 + h * half) * 2);
                  gridBase[h] = shiftIdxU16(gidxU16);
                }
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
                // The per-COLUMN scalar group-of-8 activation sum (ggml's lsum2).
                llvm::SmallVector<mlir::Value> groupSum(activationInterleave);
                for (int64_t j = 0; j < 8; ++j) {
                  int64_t k = ib * subBlockSize + grp * 8 + j;
                  llvm::SmallVector<mlir::Value> w(numHalves);
                  for (int64_t h = 0; h < numHalves; ++h)
                    w[h] = gatherByte(gridI8Ptr, gridBase[h], j);
                  for (int64_t c = cLo; c < cHi; ++c) {
                    step("act_quant_addr");
                    mlir::Value aq = i8Read(
                        al,
                        activationQuantOffset + k * activationInterleave + c);
                    groupSum[c] = j == 0
                                      ? aq
                                      : rewriter
                                            .create<emitc::AddOp>(loc, i32Type,
                                                                  groupSum[c], aq)
                                            .getResult();
                    for (int64_t h = 0; h < numHalves; ++h) {
                      mlir::Value cur =
                          rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                         subVar[c][h])
                              .getResult();
                      rewriter.create<emitc::AssignOp>(
                          loc, subVar[c][h], vwaddw(cur, vwmul(aq, w[h])));
                    }
                  }
                }
                step("group_scale_fold");
                for (int64_t c = cLo; c < cHi; ++c)
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value subV =
                        rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                       subVar[c][h])
                            .getResult();
                    mlir::Value curB =
                        rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                       sumi1Var[c][h])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(
                        loc, sumi1Var[c][h], vmaccVV(curB, ls32[gh][h], subV));
                  }
                // The DELTA term. The ls*delta product is SHARED across the 4
                // columns (a weight fact); only the group sum is per-column.
                step("group_delta_fold");
                llvm::SmallVector<mlir::Value> lsDelta(numHalves);
                for (int64_t h = 0; h < numHalves; ++h)
                  lsDelta[h] = vmulVV(ls32[gh][h], deltaVal32(bl, ib, grp, h));
                for (int64_t c = cLo; c < cHi; ++c)
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value cur2 =
                        rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                       sumi2Var[c][h])
                            .getResult();
                    rewriter.create<emitc::AssignOp>(
                        loc, sumi2Var[c][h],
                        vmaccVX(cur2, groupSum[c], lsDelta[h]));
                  }
              }
            }

            // ===== End-of-block per-column DeltaGridGroupSum fold. =====
            for (int64_t c = cLo; c < cHi; ++c)
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value dStrip = loadF16Strip(bl, h * half);
                mlir::Value dW =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfwcvtCallee,
                                   mlir::ValueRange{dStrip, vl8}, opName, role);
                mlir::Value d0 = emitOpaqueCall(
                    rewriter, loc, f32m2Type, vfmulVfCallee,
                    mlir::ValueRange{dW, aD[c], vl8}, opName, role);
                mlir::Value sumi1V =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                   sumi1Var[c][h])
                        .getResult();
                mlir::Value sumi1F =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumi1V, vl8}, opName, role);
                mlir::Value sumi2V =
                    rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                   sumi2Var[c][h])
                        .getResult();
                mlir::Value sumi2F =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                   mlir::ValueRange{sumi2V, vl8}, opName, role);
                step("iq1m_delta_scale");
                mlir::Value deltaScaled = emitOpaqueCallBuilt(
                    rewriter, loc, f32m2Type, vfmulVfCallee, opName, role,
                    [&](mlir::OpBuilder &b,
                        mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                      mlir::Value eighth =
                          rewriter.create<emitc::LiteralOp>(loc, floatType,
                                                            "0.125f")
                              .getResult();
                      return {sumi2F, eighth, vl8};
                    },
                    llvm::StringRef("eighth_scale"));
                mlir::Value inner = emitOpaqueCall(
                    rewriter, loc, f32m2Type, vfaddVVCallee,
                    mlir::ValueRange{sumi1F, deltaScaled, vl8}, opName, role,
                    llvm::StringRef("delta_grid_inner"));
                mlir::Value curF =
                    rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                        .getResult();
                mlir::Value nextF =
                    emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccVVCallee,
                                   mlir::ValueRange{curF, inner, d0, vl8},
                                   opName, role);
                rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], nextF);
              }
          }

          // Per-column per-strip store: s + (y*4+c)*bs + x*16 + h*half. NO
          // trailing 0.125 (it rode the per-block sumi2 term).
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

    // RESULT-LESS: the lane-wise vector store is the sole sink.
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
