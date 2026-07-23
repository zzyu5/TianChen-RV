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

// Base grid/sign-plane repack artifact consumers and typed plan tables.

// The ggml iq2_xxs x q8_K 16x1-REPACKED GEVM (decode). The FIRST SUPER-BLOCK GRID
// CODEBOOK + SIGN-PLANE block-as-lane repack. iq2_xxs is a QK_K=256 super-block of 8
// sub-blocks of 32; each sub-block is 4 GRID groups of 8. The block_iq2_xxsx16 repack
// stores, per (sub-block, group, column): a raw 8-bit grid INDEX (into the fixed
// 256-entry iq2xxs_grid) + a raw 7-bit sign SELECTOR (into the derived signs64 +-1
// plane); plus a per-(sub-block, column) int8 ls scale + the per-column fp16 d. The
// weight is decoded LANE-WISE (lane = column) by TWO REAL memory GATHERS: grid byte
// grid[index*8+j] (vluxei16_v_i8 over the flat int8 grid table) and sign byte
// signs64[sel*8+j] (vluxei16_v_i8 over the +-1 plane), folded via vmul-onto-grid. The
// 32-element sub-block dot is accumulated in i32 (grid*sign*q8 overflows i16), the
// unsigned ls scale weights it via vmacc_vv_i32, the fp16*fp32 fold has NO min, and
// the per-column output is scaled by 0.125 before the store. The grid + signs64 tables
// are FIXED canonical decls (emitIQ2XXSCanonicalGridTableDecl / ...Signs64TableDecl).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGridGemvBodyIq2Xxs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value columnCount, mlir::Value avlArg, mlir::Type sizeType,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef coreLmul,
    int64_t qk, int64_t weightStride, int64_t activationStride,
    int64_t gridIdxOffset, int64_t lsOffset, int64_t signOffset,
    int64_t activationQuantOffset, int64_t nSubblocks, int64_t weightInterleave,
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

    // block_iq2_xxsx16 repack facts (I4 mirror) are PARAMETERS now (the loop body op's
    // pinned attrs + the grid core brick's grid/ls/sign byte offsets + n_subblocks, read
    // by the grid branch of emitTypedRepackGemvLoopBody and passed in): qk (256),
    // weightStride (1184), activationStride (292), gridIdxOffset (160), lsOffset (32),
    // signOffset (672), activationQuantOffset (4), nSubblocks (8), weightInterleave (16),
    // half.
    int64_t numHalves = weightInterleave / half;
    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t numGroups = subBlockSize / 8;            // 4 grid entries / sub-block

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemv-iq2_xxs output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The FIXED canonical GRID-of-8 table + the DERIVED signs64 +-1 sign plane,
    // emitted ONCE (the SHARED byte-exact anchors the block-dot iq2_xxs path uses).
    emitIQ2XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XXSCanonicalSigns64TableDecl(rewriter, loc);
    // const int8_t *grid8 = (const int8_t *)weft_iq2xxs_grid;  (byte view for the
    // per-lane grid GATHER). const int8_t *signs8 = weft_iq2xxs_signs64;
    mlir::Value gridArrName = rewriter.create<emitc::LiteralOp>(
        loc, i64PtrType, "weft_iq2xxs_grid");
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();
    mlir::Value signsI8Ptr = rewriter.create<emitc::LiteralOp>(
        loc, i8PtrType, "weft_iq2xxs_signs64");

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // ---- typed sub-load / gather helpers ----
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
    // ls scale strip: int8 [1,31] widened to i32 (sext == zext for positive).
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    // The per-(sub-block, group) index base = vsll(vzext(idxU8), 3) -- the u16 grid/
    // sign byte offset index*8, to which the per-value j (0..7) is added below.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    auto idxBaseU16 = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value w16 = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("gather_index_widen"));
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {w16, three, vl8};
          });
    };
    // The REAL per-lane vluxei16 GATHER: gather byte (base + j) per lane from the
    // int8 table. The fractional mf2 anchor forbids a register vrgather over the
    // 2048-byte grid / 1024-byte sign plane.
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
    // gs = vmul_vv_i8(grid, signs) -- sign folded ONTO the grid (grid byte <= 43,
    // so grid*sign fits i8; folding onto q8 would wrap on -128).
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

        // Per-strip i32 block accumulator (ls-weighted sub-block dots).
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
          // ===== Per grid-GROUP loop grp in 0..3 (4 grid entries / sub-block). ==
          for (int64_t grp = 0; grp < numGroups; ++grp) {
            step("grid_sign_group");
            // The per-column grid INDEX + sign SELECTOR strips (u8) and their u16
            // gather bases (index*8), per strip.
            llvm::SmallVector<mlir::Value> gridBase(numHalves), signBase(numHalves);
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value gidxU8 = loadU8Strip(
                  bl, gridIdxOffset + (ib * numGroups + grp) * 16 + h * half);
              mlir::Value sselU8 = loadU8Strip(
                  bl, signOffset + (ib * numGroups + grp) * 16 + h * half);
              gridBase[h] = idxBaseU16(gidxU8);
              signBase[h] = idxBaseU16(sselU8);
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

      // Per-strip store: out[x*16 + h*half] = 0.125f * sumf_h.  (the iq2_xxs 1/8.)
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
    // has NO result -- the lane-wise vector store is the sole sink.
    return mlir::success();
  }

// The ggml iq2_xxs x q8_K 16x1-REPACKED PREFILL GEMM. The iq2_xxs prefill sibling of
// emitRepackGridGemvBodyIq2Xxs: the SAME real grid GATHER + sign-plane GATHER + ls scale +
// 0.125 fold, with the grid+sign weight decode AMORTIZED across the 4 interleaved
// block_q8_Kx4 activation columns (4 fp32 d at +0, interleaved int8 quants at +16 as
// pos*4+c). Every weight byte is read once per group.
mlir::LogicalResult VariantToEmitCFunc::emitRepackGridGemmBodyIq2Xxs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t gridIdxOffset,
    int64_t lsOffset, int64_t signOffset, int64_t activationQuantOffset,
    int64_t nSubblocks, int64_t weightInterleave, int64_t activationInterleave,
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
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // block_iq2_xxsx16 repack facts (I4 mirror) are PARAMETERS now (the loop body op's
    // pinned attrs + the grid core brick's grid/ls/sign byte offsets + n_subblocks, read
    // by the grid branch of emitTypedRepackGemmLoopBody and passed in): qk (256),
    // weightStride (1184), activationStride (1168), gridIdxOffset (160), lsOffset (32),
    // signOffset (672), activationQuantOffset (16), nSubblocks (8), weightInterleave (16),
    // activationInterleave (4), half.
    int64_t numHalves = weightInterleave / half;
    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t numGroups = subBlockSize / 8;            // 4
    int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-iq2_xxs output not pointer");

    mlir::Value vl8 = sizeLit(half);

    emitIQ2XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XXSCanonicalSigns64TableDecl(rewriter, loc);
    mlir::Value gridArrName = rewriter.create<emitc::LiteralOp>(
        loc, i64PtrType, "weft_iq2xxs_grid");
    mlir::Value gridI8Ptr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrName).getResult();
    mlir::Value signsI8Ptr = rewriter.create<emitc::LiteralOp>(
        loc, i8PtrType, "weft_iq2xxs_signs64");

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
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    auto idxBaseU16 = [&](mlir::Value idxU8) -> mlir::Value {
      mlir::Value w16 = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("gather_index_widen"));
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value three =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "3")
                    .getResult();
            return {w16, three, vl8};
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

            // Per-column activation super-block scale d_y_c (4 fp32 at 0,4,8,12).
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

            // Per-column i32 block accumulator (ls-weighted sub-block dots).
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
              step("subblock_ls_scale");
              llvm::SmallVector<mlir::Value> ls32(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                ls32[h] = lsScale32(bl, ib, h);
              // Per-column i32 sub-block dot accumulator per strip.
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
              // ===== Per grid-GROUP loop grp in 0..3. SHARED weight decode. =====
              for (int64_t grp = 0; grp < numGroups; ++grp) {
                step("grid_sign_group");
                llvm::SmallVector<mlir::Value> gridBase(numHalves),
                    signBase(numHalves);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value gidxU8 = loadU8Strip(
                      bl, gridIdxOffset + (ib * numGroups + grp) * 16 + h * half);
                  mlir::Value sselU8 = loadU8Strip(
                      bl, signOffset + (ib * numGroups + grp) * 16 + h * half);
                  gridBase[h] = idxBaseU16(gidxU8);
                  signBase[h] = idxBaseU16(sselU8);
                }
                for (int64_t j = 0; j < 8; ++j) {
                  int64_t k = ib * subBlockSize + grp * 8 + j;
                  // SHARED grid*sign weight per strip (reused across columns).
                  llvm::SmallVector<mlir::Value> w(numHalves);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value gridV = gatherByte(gridI8Ptr, gridBase[h], j);
                    mlir::Value signV = gatherByte(signsI8Ptr, signBase[h], j);
                    w[h] = signFold(gridV, signV);
                  }
                  for (int64_t c = cLo; c < cHi; ++c) {
                    step("act_quant_addr");
                    mlir::Value aq = i8Read(
                        al, activationQuantOffset + k * activationInterleave + c);
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
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[c][h])
                          .getResult();
                  mlir::Value curB =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     sumiVar[c][h])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c][h], vmaccVV(curB, ls32[h], subV));
                }
            }

            // ===== End-of-block per-column fold: sumf += cvt(sumi)*(d_x*d_y). ===
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
    // has NO result -- the lane-wise vector store is the sole sink.
    return mlir::success();
  }

// ============================================================================
// iq2_xs / iq2_s DUAL-scale GRID+SIGN 16x1-REPACKED shared bodies.
//
// These reuse the iq2_xxs block-as-lane scaffold VERBATIM (the same real vluxei16
// grid GATHER + vluxei16 sign-plane GATHER + vmul-onto-grid fold + i32-accumulator
// dot + fp16*fp32 no-min fold + 0.125 store) with exactly TWO structural deltas:
//   (a) the grid INDEX is a u16 strip (9-bit for iq2_xs, 10-bit for iq2_s) loaded
//       DIRECTLY with vle16 + vsll (NO vzext -- the index cannot live in a byte);
//   (b) each 32-lane sub-block is split into TWO ls-weighted group-halves (ls1 over
//       groups 0-1, ls2 over groups 2-3), each with its own i32 dot -> vmacc.
// `variant` picks the FIXED canonical tables (iq2_xs: 512-grid + ksigns signs64;
// iq2_s: 1024-grid + explicit signs256) emitted ONCE as static const decls.
// ============================================================================
// C4a: the ONE plan -> emit-period table-decl dispatch. Keyed on the registry row's
// decode_model; the DECL bodies (ggml's exact literals) stay in the Conversion layer.
// A registered plan with no decl arm FAILS here rather than emitting a reference to an
// undeclared table.
// The ggml iq3_xxs x q8_K 16x1-REPACKED block-as-lane GEVM (decode) -- C4a-4, and the
// FIRST leaf built around a grid entry that covers only HALF an activation group.
//
// WHY THIS IS A LEAF AND NOT A PARAMETER OF emitRepackGridGemvBodyIq2Xxs. On every axis
// the plan records, iq3_xxs and iq2_xxs agree: GridLsArity::Single, GridSignPlane::
// Signs64, GridFoldArith::SignScaleStore, gridEntryCount 256. They agree in ggml too --
// ggml_vec_dot_iq3_xxs_q8_K's `ls = 2*(aux32 >> 28) + 1`, its `ksigns_iq2xs[(aux32 >>
// 7*l) & 127]` selector tested by `kmask_iq2xs[j]`, and its single-i32 `bsum += sumi*ls`
// / `sumf += d*bsum` chain are iq2_xxs's, instruction for instruction. The ONE axis they
// differ on is GridEntryWidth, and it is the one that reshapes the nest:
//
//   iq2_xxs (I64x8):  grid  = iq2xxs_grid + aux8[l];        for j<8: grid[j]*q8[j]
//   iq3_xxs (I32x4):  grid1 = iq3xxs_grid + q3[2*l+0];      for j<4: grid1[j]*q8[j+0]
//                     grid2 = iq3xxs_grid + q3[2*l+1];               grid2[j]*q8[j+4]
//
// A uint32 entry carries 4 grid bytes, so an 8-lane group needs TWO entries and the
// activation range SPLITS: lanes 0-3 read gridBase1 at byte j, lanes 4-7 read gridBase2
// at byte j-4. The iq2_xxs nest hoists ONE base per group and walks j = 0..7 against it;
// feeding this row through it would not fail -- it would gather bytes 4..7 of a 4-byte
// entry, i.e. the NEXT entry's bytes, for the upper half of every group, and produce
// confidently wrong numbers. That is why the dispatcher keys the leaf off
// entryWidth == I32x4 BEFORE it tests ls arity (GridDecodePlan.h's selection order).
//
// THREE further consequences, all of them structural rather than constant-valued:
//   (1) TWO index strips at different rates. The grid-index strip carries 8 raw q3 bytes
//       per sub-block (gidx[8][8][16], 1024 B) while the sign-selector strip carries 4
//       (ssel[8][4][16], 512 B) -- the sign selector is still ONE per 8-lane group, since
//       kmask_iq2xs[j+0] covers the grid1 lanes and kmask_iq2xs[j+4] the grid2 lanes of
//       the SAME selector byte. So the two strips index at different rates for the first
//       time in this family.
//   (2) TWO gather shifts. The grid gather shifts an index by 2 (idx*4, the uint32 entry)
//       but the sign gather still shifts by 3 (sel*8, the DERIVED signs64 plane's 8 +-1
//       bytes per selector). In every I64x8 leaf both are 3 and one helper serves both;
//       here they must not be the same helper.
//   (3) The store constant is ggml's 0.25f, not 0.125f -- carried as plan data
//       (storeScaleLiteral) with a matching DERIVED role marker, so the shipped step
//       comment cannot name a constant it is not attached to.
//
// Everything else -- the vluxei16 grid/sign gather, the vmul-onto-grid sign fold (iq3's
// grid bytes reach 62 < 128, so grid*sign still fits i8 exactly as iq2_xxs's <= 43 do),
// the i32-accumulator dot, the ls vmacc, the fp16*fp32 no-min fold -- is iq2_xxs's,
// unchanged. RESULT-LESS (no monolith token).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvGridDualEntryQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase,
    mlir::Value activationBase, mlir::Value output, mlir::Value columnCount,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t gridIdxOffset,
    int64_t lsOffset, int64_t signOffset, int64_t activationQuantOffset,
    int64_t nSubblocks, int64_t weightInterleave, int64_t half) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The store-side scale marker is DERIVED from the plan's literal and the mapping is
    // CLOSED: an unregistered literal yields an empty marker and we refuse to emit rather
    // than stamp a comment naming the wrong constant.
    llvm::StringRef storeScaleRole = weft::gridStoreScaleRole(plan.storeScaleLiteral);
    if (plan.storeScaleLiteral.empty() || storeScaleRole.empty())
      return rewriter.notifyMatchFailure(
          loc, "the dual-entry repack GEVM leaf requires a SignScaleStore plan whose "
               "storeScaleLiteral has a registered store-scale role marker");
    if (plan.entryWidth != weft::GridEntryWidth::I32x4)
      return rewriter.notifyMatchFailure(
          loc, "the dual-entry repack GEVM leaf lowers ONLY I32x4 rows (a "
               "4-byte grid entry covering half an 8-element group); an I64x8 row "
               "belongs on the single-base iq2_xxs leaf");

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
    // The I32x4 grids are `static const uint32_t weft_iq3xxs_grid[256]` /
    // `weft_iq3s_grid[512]` -- the ONLY registered rows whose grid decl is NOT an int64
    // array.
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The repack facts are PARAMETERS (the loop body op's pinned attrs + the grid core
    // brick's grid/ls/sign byte offsets + n_subblocks): qk (256), weightStride (1696
    // iq3_xxs / 2720 iq3_s), activationStride (292), gridIdxOffset (160), lsOffset (32),
    // signOffset (1184 / 2208), activationQuantOffset (4), nSubblocks (8),
    // weightInterleave (16), half.
    int64_t numHalves = weightInterleave / half;
    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t numGroups = subBlockSize / 8;            // 4 sign selectors / sub-block
    // The I32x4 fact, DERIVED (never a second field that could disagree with the width):
    // 2 grid entries per 8-lane group => 8 grid indices per sub-block.
    int64_t entriesPerGroup = weft::gridEntriesPerGroup(plan.entryWidth);
    int64_t gridIdxPerSubblock = numGroups * entriesPerGroup;
    int64_t gridShift = weft::gridEntryByteShift(plan.entryWidth);   // 2 (idx*4)
    int64_t lanesPerEntry = 8 / entriesPerGroup;                     // 4
    // The grid-INDEX strip's lane width, DERIVED from the entry COUNT (C4a-5): iq3_xxs's
    // 256-entry grid indexes with a byte, iq3_s's 512-entry grid needs a 9-bit index and
    // therefore a u16 lane. This is a PARAMETER of this leaf, not a second leaf: it swaps
    // the index LOAD (vle8+vzext vs vle16) and nothing else -- the dual-base nest, the
    // activation split, the sign gather, the accumulator arity and the fold are identical
    // for both rows. See weft::gridIndexStripIsU16.
    bool gridIdxU16 = weft::gridIndexStripIsU16(plan.gridEntryCount);
    int64_t gridIdxLaneBytes = gridIdxU16 ? 2 : 1;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemv-iq3_xxs output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The FIXED uint32 grid table + the DERIVED signs64 +-1 sign plane (iq2_xxs's, which
    // is ggml's ksigns_iq2xs -- the table iq3_xxs's own vec_dot reads by that name).
    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "iq3_xxs repack GEVM: the plan names tables with no decl emitter");
    // const int8_t *grid8 = (const int8_t *)weft_iq3xxs_grid;  (byte view for the
    // per-lane grid GATHER). const int8_t *signs8 = weft_iq2xxs_signs64;
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, u32PtrType, plan.gridArrayName);
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

    // ---- typed sub-load / gather helpers ----
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
    // ls scale strip: int8 [1,31] widened to i32 (sext == zext for positive).
    std::string vsextCallee = ("__riscv_vsext_vf4_i32" + l32).str();
    auto lsScale32 = [&](mlir::Value bl, int64_t ib, int64_t h) -> mlir::Value {
      mlir::Value lsI8 = loadI8Strip(bl, lsOffset + ib * 16 + h * half);
      return emitOpaqueCall(rewriter, loc, i32m2Type, vsextCallee,
                            mlir::ValueRange{lsI8, vl8}, opName, role,
                            llvm::StringRef("subblock_scale_widen"));
    };
    // u16 grid-index strip (C4a-5, the iq3_s path): a 9-bit index cannot live in a byte,
    // so it is a uint16 lane loaded DIRECTLY (vle16 -- NO vzext). The SAME shape the
    // iq2_xs / iq2_s / iq1_s leaves already use for their 9/10/11-bit indices; only the
    // 256-entry rows (iq2_xxs, iq3_xxs) index with a byte. NOT emitted for a u8 row --
    // gridIdxU16 gates every use, so iq3_xxs's emitted C is untouched.
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
    // The per-(sub-block, group) gather base = vsll(vzext(idxU8), shift). The GRID and
    // SIGN shifts DIFFER on an I32x4 row -- 2 for the uint32 grid entry (idx*4), 3 for
    // the signs64 selector (sel*8, 8 +-1 bytes each). The I64x8 leaves use one helper for
    // both because there both are 3; conflating them here would gather the wrong plane.
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    auto idxBaseU16 = [&](mlir::Value idxU8, int64_t shift) -> mlir::Value {
      mlir::Value w16 = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("gather_index_widen"));
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sh =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                  std::to_string(shift))
                    .getResult();
            return {w16, sh, vl8};
          });
    };
    // The u16-index counterpart: the lane is ALREADY 16-bit, so the base is vsll ONLY
    // (no vzext). Same shift, same result shape -- the two differ exactly in whether the
    // index arrived widened. (C4a-5; the iq2_xs / iq2_s / iq1_s leaves have this same
    // pair split the same way.)
    auto gridBaseFromStrip = [&](mlir::Value idx, int64_t shift) -> mlir::Value {
      if (!gridIdxU16)
        return idxBaseU16(idx, shift);
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sh =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                  std::to_string(shift))
                    .getResult();
            return {idx, sh, vl8};
          });
    };
    // The REAL per-lane vluxei16 GATHER: gather byte (base + j) per lane from the
    // int8 table. The fractional mf2 anchor forbids a register vrgather over the
    // 1024-byte grid / 2048-byte grid / 1024- or 2048-byte sign plane.
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
    // gs = vmul_vv_i8(grid, signs) -- sign folded ONTO the grid. iq3_xxs's grid bytes
    // reach 62 (0x3e), so grid*sign fits i8; folding onto q8 would wrap on -128.
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

        // Per-strip i32 block accumulator (ls-weighted sub-block dots).
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
          // ===== Per grid-GROUP loop grp in 0..3. =====
          for (int64_t grp = 0; grp < numGroups; ++grp) {
            step("grid_sign_group");
            // THE DUAL-ENTRY NEST. gridBase[e][h] for e in {0,1}: TWO gather bases per
            // 8-lane group, from TWO CONSECUTIVE grid-index strip entries (ggml's
            // q3[2*l+0] / q3[2*l+1]), each shifted by 2 (uint32 entry). The sign
            // selector stays ONE per group, shifted by 3 (signs64's 8 bytes/selector).
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> gridBase(
                entriesPerGroup);
            llvm::SmallVector<mlir::Value> signBase(numHalves);
            for (int64_t e = 0; e < entriesPerGroup; ++e)
              for (int64_t h = 0; h < numHalves; ++h) {
                // The strip's LANE WIDTH is the row's (u8 for a 256-entry grid, u16 for
                // iq3_s's 512); the byte offset scales with it. Everything downstream --
                // base shift, gather, split -- is identical for both.
                int64_t gidxByteOff =
                    gridIdxOffset +
                    ((ib * gridIdxPerSubblock + grp * entriesPerGroup + e) * 16 +
                     h * half) *
                        gridIdxLaneBytes;
                mlir::Value gidx = gridIdxU16 ? loadU16Strip(bl, gidxByteOff)
                                              : loadU8Strip(bl, gidxByteOff);
                gridBase[e].push_back(gridBaseFromStrip(gidx, gridShift));
              }
            for (int64_t h = 0; h < numHalves; ++h) {
              mlir::Value sselU8 = loadU8Strip(
                  bl, signOffset + (ib * numGroups + grp) * 16 + h * half);
              signBase[h] = idxBaseU16(sselU8, 3);
            }
            for (int64_t j = 0; j < 8; ++j) {
              step("act_quant_addr");
              int64_t k = ib * subBlockSize + grp * 8 + j;
              mlir::Value aq = i8Read(al, activationQuantOffset + k);
              // THE ACTIVATION-RANGE SPLIT: lane j reads entry (j / 4) at byte (j % 4).
              // ggml: `for j<4 { grid1[j]*q8[j+0]; grid2[j]*q8[j+4]; }`. The sign plane
              // is NOT split -- kmask_iq2xs[j] indexes all 8 bits of the ONE selector.
              int64_t e = j / lanesPerEntry;
              int64_t gj = j % lanesPerEntry;
              for (int64_t h = 0; h < numHalves; ++h) {
                mlir::Value gridV = gatherByte(gridI8Ptr, gridBase[e][h], gj);
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

      // Per-strip store: out[x*16 + h*half] = 0.25f * sumf_h.  (ggml's iq3_xxs 1/4,
      // carried as plan data -- the marker below is DERIVED from it.)
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
              mlir::Value q =
                  rewriter.create<emitc::LiteralOp>(loc, floatType,
                                                    plan.storeScaleLiteral)
                      .getResult();
              return {sumfVal, q, vl8};
            },
            storeScaleRole);
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, scaled, vl8}, opName, role);
      }
    }

    // RESULT-LESS (no monolith token): the front-door typed_repack_gemv_loop_body region
    // has NO result -- the lane-wise vector store is the sole sink.
    return mlir::success();
  }


// The ggml iq3_xxs x q8_K 16x1-REPACKED PREFILL GEMM (C4a-4). The iq3_xxs prefill sibling
// of emitRepackGemvGridDualEntryQ8K: the SAME DUAL-ENTRY grid GATHER (two uint32 gridBases per
// 8-lane group, activation range split 0-3 / 4-7) + single sign-plane GATHER per group +
// ls scale + 0.25f store fold, with the grid+sign weight decode AMORTIZED across the 4
// interleaved block_q8_Kx4 activation columns (4 fp32 d at +0, interleaved int8 quants at
// +16 as pos*4+c). Every weight byte is read once per group.
//
// See emitRepackGemvGridDualEntryQ8K for why the dual-entry nest is a LEAF rather than a
// parameter of the iq2_xxs leaves; this file is that argument's prefill half, and the
// amortization does not change it -- the split is in the WEIGHT decode, which is exactly
// the part the 4 columns share. Ships PLAIN (untiled), like its iq2_xxs sibling.
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmGridDualEntryQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan, mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value rowCount, mlir::Value columnCount, mlir::Value outputRowStride,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef coreLmul, int64_t qk,
    int64_t weightStride, int64_t activationStride, int64_t gridIdxOffset,
    int64_t lsOffset, int64_t signOffset, int64_t activationQuantOffset,
    int64_t nSubblocks, int64_t weightInterleave, int64_t activationInterleave,
    int64_t half, bool colGroupOuter) const {
    mlir::MLIRContext *ctx = rewriter.getContext();

    // Same CLOSED store-scale marker + entry-width guards as the GEVM leaf: refuse to
    // emit rather than stamp a comment naming a constant it is not attached to, and
    // refuse an I64x8 row outright (it belongs on the single-base iq2_xxs leaf).
    llvm::StringRef storeScaleRole = weft::gridStoreScaleRole(plan.storeScaleLiteral);
    if (plan.storeScaleLiteral.empty() || storeScaleRole.empty())
      return rewriter.notifyMatchFailure(
          loc, "the dual-entry repack GEMM leaf requires a SignScaleStore plan whose "
               "storeScaleLiteral has a registered store-scale role marker");
    if (plan.entryWidth != weft::GridEntryWidth::I32x4)
      return rewriter.notifyMatchFailure(
          loc, "the dual-entry repack GEMM leaf lowers ONLY I32x4 rows");

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
    // The I32x4 grid decls are `static const uint32_t weft_iq3xxs_grid[256]` /
    // `weft_iq3s_grid[512]` -- the ONLY registered rows whose grid is not an int64 array.
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // block_iq3_xxsx16 repack facts are PARAMETERS (the loop body op's pinned attrs + the
    // grid core brick's grid/ls/sign byte offsets + n_subblocks, read by the grid branch
    // of emitTypedRepackGemmLoopBody and passed in): qk (256), weightStride (1696),
    // activationStride (1168), gridIdxOffset (160), lsOffset (32), signOffset (1184),
    // activationQuantOffset (16), nSubblocks (8), weightInterleave (16),
    // activationInterleave (4), half. The strip that grew vs iq2_xxs's 1184 stride is the
    // grid index (512 B -> 1024 B): TWO u8 indices per group instead of one.
    int64_t numHalves = weightInterleave / half;
    int64_t subBlockSize = qk / nSubblocks;          // 32
    int64_t numGroups = subBlockSize / 8;            // 4 sign selectors / sub-block
    // The I32x4 facts, DERIVED from the entry width (never a second field that could
    // disagree with it): 2 grid entries per 8-lane group => 8 grid indices per sub-block,
    // gather shift 2 (idx*4), 4 activation lanes per entry.
    int64_t entriesPerGroup = weft::gridEntriesPerGroup(plan.entryWidth);
    int64_t gridIdxPerSubblock = numGroups * entriesPerGroup;
    int64_t gridShift = weft::gridEntryByteShift(plan.entryWidth);
    int64_t lanesPerEntry = 8 / entriesPerGroup;
    // The grid-INDEX strip lane width, DERIVED from the entry COUNT (C4a-5): iq3_xxs's
    // 256-entry grid indexes with a byte, iq3_s's 512-entry grid needs 9 bits => u16.
    // A leaf PARAMETER (it swaps the index LOAD only), not a leaf key -- see
    // weft::gridIndexStripIsU16 and the GEVM sibling.
    bool gridIdxU16 = weft::gridIndexStripIsU16(plan.gridEntryCount);
    int64_t gridIdxLaneBytes = gridIdxU16 ? 2 : 1;
    int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(loc,
                                         "repack-gemm-iq3_xxs output not pointer");

    mlir::Value vl8 = sizeLit(half);

    // The FIXED uint32 grid + the DERIVED signs64 +-1 plane the PLAN names (the sign
    // table is iq2_xxs's -- ggml's iq3_xxs vec_dot reads ksigns_iq2xs by that name).
    if (mlir::failed(emitGridDecodePlanTableDecls(rewriter, loc, plan)))
      return rewriter.notifyMatchFailure(
          loc, "dual-entry repack GEMM: the plan names tables with no decl emitter");
    mlir::Value gridArrName =
        rewriter.create<emitc::LiteralOp>(loc, u32PtrType, plan.gridArrayName);
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
    // u16 grid-index strip (C4a-5, the iq3_s path): a 9-bit index cannot live in a byte,
    // so it is a uint16 lane loaded DIRECTLY (vle16 -- NO vzext). gridIdxU16 gates every
    // use, so iq3_xxs's emitted C is untouched.
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
    std::string vzextCallee = ("__riscv_vzext_vf2_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    // GRID and SIGN gather shifts DIFFER on an I32x4 row (2 for the uint32 grid entry,
    // 3 for the signs64 selector's 8 +-1 bytes). The I64x8 leaves use one helper because
    // there both are 3; conflating them here would gather the wrong plane.
    auto idxBaseU16 = [&](mlir::Value idxU8, int64_t shift) -> mlir::Value {
      mlir::Value w16 = emitOpaqueCall(rewriter, loc, u16m1Type, vzextCallee,
                                       mlir::ValueRange{idxU8, vl8}, opName, role,
                                       llvm::StringRef("gather_index_widen"));
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sh =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                  std::to_string(shift))
                    .getResult();
            return {w16, sh, vl8};
          });
    };
    // The u16-index counterpart: the lane is ALREADY 16-bit, so the base is vsll ONLY
    // (no vzext). Same shift, same result shape -- the two differ exactly in whether the
    // index arrived widened. (C4a-5; see the GEVM sibling.)
    auto gridBaseFromStrip = [&](mlir::Value idx, int64_t shift) -> mlir::Value {
      if (!gridIdxU16)
        return idxBaseU16(idx, shift);
      return emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vsllU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sh =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type,
                                                  std::to_string(shift))
                    .getResult();
            return {idx, sh, vl8};
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

            // Per-column activation super-block scale d_y_c (4 fp32 at 0,4,8,12).
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

            // Per-column i32 block accumulator (ls-weighted sub-block dots).
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
              step("subblock_ls_scale");
              llvm::SmallVector<mlir::Value> ls32(numHalves);
              for (int64_t h = 0; h < numHalves; ++h)
                ls32[h] = lsScale32(bl, ib, h);
              // Per-column i32 sub-block dot accumulator per strip.
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
              // ===== Per grid-GROUP loop grp in 0..3. SHARED weight decode. =====
              for (int64_t grp = 0; grp < numGroups; ++grp) {
                step("grid_sign_group");
                // THE DUAL-ENTRY NEST. gridBase[e][h] for e in {0,1}: TWO gather bases
                // per 8-lane group from TWO CONSECUTIVE grid-index strip entries (ggml's
                // q3[2*l+0] / q3[2*l+1]), each shifted by 2. The sign selector stays ONE
                // per group (shift 3): kmask_iq2xs[j+0] covers the grid1 lanes and
                // kmask_iq2xs[j+4] the grid2 lanes of the SAME selector byte.
                llvm::SmallVector<llvm::SmallVector<mlir::Value>> gridBase(
                    entriesPerGroup);
                llvm::SmallVector<mlir::Value> signBase(numHalves);
                for (int64_t e = 0; e < entriesPerGroup; ++e)
                  for (int64_t h = 0; h < numHalves; ++h) {
                    // Strip LANE WIDTH is the row's (u8 for a 256-entry grid, u16 for
                    // iq3_s's 512); the byte offset scales with it, everything
                    // downstream is identical. See the GEVM sibling.
                    int64_t gidxByteOff =
                        gridIdxOffset +
                        ((ib * gridIdxPerSubblock + grp * entriesPerGroup + e) * 16 +
                         h * half) *
                            gridIdxLaneBytes;
                    mlir::Value gidx = gridIdxU16 ? loadU16Strip(bl, gidxByteOff)
                                                  : loadU8Strip(bl, gidxByteOff);
                    gridBase[e].push_back(gridBaseFromStrip(gidx, gridShift));
                  }
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value sselU8 = loadU8Strip(
                      bl, signOffset + (ib * numGroups + grp) * 16 + h * half);
                  signBase[h] = idxBaseU16(sselU8, 3);
                }
                for (int64_t j = 0; j < 8; ++j) {
                  int64_t k = ib * subBlockSize + grp * 8 + j;
                  // THE ACTIVATION-RANGE SPLIT: lane j reads entry (j / 4) at byte
                  // (j % 4). The sign plane is NOT split.
                  int64_t e = j / lanesPerEntry;
                  int64_t gj = j % lanesPerEntry;
                  // SHARED grid*sign weight per strip (reused across columns).
                  llvm::SmallVector<mlir::Value> w(numHalves);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value gridV = gatherByte(gridI8Ptr, gridBase[e][h], gj);
                    mlir::Value signV = gatherByte(signsI8Ptr, signBase[h], j);
                    w[h] = signFold(gridV, signV);
                  }
                  for (int64_t c = cLo; c < cHi; ++c) {
                    step("act_quant_addr");
                    mlir::Value aq = i8Read(
                        al, activationQuantOffset + k * activationInterleave + c);
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
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type, subVar[c][h])
                          .getResult();
                  mlir::Value curB =
                      rewriter.create<emitc::LoadOp>(loc, i32m2Type,
                                                     sumiVar[c][h])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(
                      loc, sumiVar[c][h], vmaccVV(curB, ls32[h], subV));
                }
            }

            // ===== End-of-block per-column fold: sumf += cvt(sumi)*(d_x*d_y). ===
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

          // Per-column per-strip store: s + (y*4+c)*bs + x*16 + h*half, x 0.25f.
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
                  storeScaleRole);
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
    // has NO result -- the lane-wise vector store is the sole sink.
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitGridDecodePlanTableDecls(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const weft::GridDecodePlan &plan) const {
  if (plan.decodeModel == "iq2_xxs") {
    emitIQ2XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XXSCanonicalSigns64TableDecl(rewriter, loc);
    return mlir::success();
  }
  if (plan.decodeModel == "iq2_xs") {
    emitIQ2XSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XSCanonicalSigns64TableDecl(rewriter, loc);
    return mlir::success();
  }
  if (plan.decodeModel == "iq2_s") {
    emitIQ2SCanonicalGridTableDecl(rewriter, loc);
    emitIQ2SCanonicalSigns256TableDecl(rewriter, loc);
    return mlir::success();
  }
  // iq1_s (C4a-2): a TernaryDelta row emits the 2048-entry ternary grid and NO
  // sign table -- there is none to emit (plan.signArrayName is empty by
  // construction). This reuses the SAME emitIQ1SCanonicalGridTableDecl the iq1_s
  // BLOCK-DOT path already emits from the canonical kIQ1SGrid, so the repack and
  // block-dot paths cannot disagree about the grid literals.
  if (plan.decodeModel == "iq1_s") {
    emitIQ1SCanonicalGridTableDecl(rewriter, loc);
    return mlir::success();
  }
  // iq1_m (C4a-3): likewise a TernaryDelta row -- the 2048-entry ternary grid and NO
  // sign table. Reuses the SAME emitIQ1MCanonicalGridTableDecl the iq1_m BLOCK-DOT
  // path already emits from the canonical kIQ1MGrid, so the repack and block-dot
  // paths cannot disagree about the grid literals. (ggml's own
  // ggml_vec_dot_iq1_m_q8_K indexes `iq1s_grid`; the decl is named per-format here
  // only because these are function-scoped emit-period decls.)
  if (plan.decodeModel == "iq1_m") {
    emitIQ1MCanonicalGridTableDecl(rewriter, loc);
    return mlir::success();
  }
  // iq3_xxs (C4a-4): the ONLY row whose grid decl is a uint32 array
  // (`static const uint32_t weft_iq3xxs_grid[256]`, ggml's iq3xxs_grid verbatim) --
  // reusing the SAME emitIQ3XXSCanonicalGridTableDecl the iq3_xxs BLOCK-DOT path already
  // emits from the canonical kIQ3XXSGrid, so the repack and block-dot paths cannot
  // disagree about the grid literals.
  //
  // Its sign plane is the DERIVED signs64 +-1 plane -- and it is LITERALLY iq2_xxs's, not
  // a lookalike. ggml's ggml_vec_dot_iq3_xxs_q8_K reads `ksigns_iq2xs` BY THAT NAME (the
  // same table its iq2_xxs sibling reads), so the plan names weft_iq2xxs_signs64 and this
  // arm emits iq2_xxs's decl unchanged. Verified rather than assumed: the repo's
  // kIQ3XXSKsigns and kIQ2XXSKsigns are both byte-identical to ggml's ksigns_iq2xs, so a
  // separate iq3-named 1024-byte decl would have been a COPY of this one, not a fact
  // about iq3_xxs. (kIQ3XXSKsigns still exists for the block-dot path, which emits the
  // raw 128-byte selector table under its own name rather than this DERIVED +-1 plane.)
  if (plan.decodeModel == "iq3_xxs") {
    emitIQ3XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XXSCanonicalSigns64TableDecl(rewriter, loc);
    return mlir::success();
  }
  // iq3_s (C4a-5): the SECOND uint32-grid row (`static const uint32_t
  // weft_iq3s_grid[512]`, ggml's iq3s_grid verbatim) -- reusing the SAME
  // emitIQ3SCanonicalGridTableDecl the iq3_s BLOCK-DOT path already emits from the
  // canonical kIQ3SGrid, so the repack and block-dot paths cannot disagree about the grid
  // literals. (kIQ3SGrid was machine-checked equal to ggml's iq3s_grid when this row
  // landed, all 512 entries.)
  //
  // Its sign plane is iq2_s's DERIVED signs256 plane, and -- unlike iq3_xxs, which reuses
  // iq2_xxs's signs64 because ggml literally reads `ksigns_iq2xs` from BOTH -- the reuse
  // here is by CONSTRUCTION rather than by shared table name: ggml's iq3_s reads no sign
  // TABLE at all, it tests the block's own explicit `signs[l]` byte with kmask_iq2xs[j].
  // The DERIVED +-1 plane that answers "byte b, lane j -> +-1" is a pure function of
  // (b, j) with no format in it, so iq2_s's 2048-byte decl IS the plane iq3_s needs, and
  // emitting a byte-identical second copy under an iq3 name would be a copy, not a fact.
  if (plan.decodeModel == "iq3_s") {
    emitIQ3SCanonicalGridTableDecl(rewriter, loc);
    emitIQ2SCanonicalSigns256TableDecl(rewriter, loc);
    return mlir::success();
  }
  return mlir::failure();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
