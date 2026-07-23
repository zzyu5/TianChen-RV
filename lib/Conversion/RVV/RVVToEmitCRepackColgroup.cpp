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

// Colgroup-tiled typed repack artifact consumers.

mlir::LogicalResult VariantToEmitCFunc::emitTypedRepackGemvColgroupTiledLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  weftrvv::TypedRepackGemvColgroupTiledLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb =
            llvm::dyn_cast<weftrvv::TypedRepackGemvColgroupTiledLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed colgroup-tiled GEVM loop body missing the op");

  if (loopBody.getFoldModel() != "kquant_dmin_bsums_min")
    return rewriter.notifyMatchFailure(
        loopBody, "colgroup-tiled GEVM plan currently supports only the q4_K "
                  "\"kquant_dmin_bsums_min\" fold (first cell)");

  weftrvv::RepackGemvKQuantCoreOp coreBrick;
  loopBody.getBody().walk(
      [&](weftrvv::RepackGemvKQuantCoreOp o) { coreBrick = o; });
  if (!coreBrick)
    return rewriter.notifyMatchFailure(
        loopBody, "colgroup-tiled GEVM loop body requires the region "
                  "weft_rvv.repack_gemv_kquant_core integer-core brick");
  if (coreBrick.getDecodeModel() != "q4_K")
    return rewriter.notifyMatchFailure(
        coreBrick, "colgroup-tiled GEVM plan's core brick must carry decode_model "
                   "\"q4_K\" (the first cell)");
  if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0))
    return rewriter.notifyMatchFailure(
        coreBrick, "the K-quant core brick's block_index must be the loop "
                   "induction variable (region arg 0)");
  if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
      coreBrick.getActivationBase() != loopBody.getActivationBase())
    return rewriter.notifyMatchFailure(
        coreBrick, "the K-quant core brick's weight/activation bases must be the "
                   "loop-body's own repacked-weight / q8_K-activation ABI buffers");

  std::optional<uint64_t> dminOff = loopBody.getWeightDminByteOffset();
  std::optional<uint64_t> scalesOff = loopBody.getWeightScalesByteOffset();
  std::optional<uint64_t> bsumsOff = loopBody.getActivationBsumsByteOffset();
  std::optional<uint64_t> nSub = loopBody.getNSubblocks();
  if (!dminOff || !scalesOff || !bsumsOff || !nSub)
    return rewriter.notifyMatchFailure(
        loopBody, "colgroup-tiled GEVM loop body requires the super-block decode "
                  "attrs weight_dmin_byte_offset / weight_scales_byte_offset / "
                  "activation_bsums_byte_offset / n_subblocks");

  mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
  mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
  mlir::Value output = valueMap.lookup(loopBody.getOutput());
  mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
  if (!weightBase || !activationBase || !output || !columnCount)
    return rewriter.notifyMatchFailure(
        loopBody, "colgroup-tiled GEVM loop ABI operand unmapped");
  llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
  // Fail-closed final capability-fact read: the
  // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
  if (!loopBody.getIntegerCoreLmul())
    return rewriter.notifyMatchFailure(
        loopBody, "repack integer core requires an explicit integer_core_lmul "
              "capability fact (front door stamps it; no silent mf2 default)");
  llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
  return emitRepackKQuantGemvColgroupTiledBodyQ4K(
      rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
      sizeType, opName, role, coreLmul,
      static_cast<int64_t>(loopBody.getQk()),
      static_cast<int64_t>(loopBody.getWeightBlockStride()),
      static_cast<int64_t>(loopBody.getActivationBlockStride()),
      static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
      static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
      static_cast<int64_t>(*dminOff), static_cast<int64_t>(*scalesOff),
      static_cast<int64_t>(*bsumsOff), static_cast<int64_t>(*nSub),
      static_cast<int64_t>(loopBody.getWeightInterleave()),
      static_cast<int64_t>(loopBody.getHalfLanes()),
      static_cast<int64_t>(loopBody.getColumnGroupTile()));
}

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmQ4_1Q8_1(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlRepackGemmQ41Q81Op gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<weftrvv::GgmlRepackGemmQ41Q81Op>(op))
        gemm = g;
    }
    if (!gemm)
      return rewriter.notifyMatchFailure(scope,
                                         "repack-gemm-q4_1 body missing op");

    mlir::Value weightBase = valueMap.lookup(gemm.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemm.getActivationBase());
    mlir::Value output = valueMap.lookup(gemm.getOutput());
    mlir::Value rowCount = valueMap.lookup(gemm.getRowCount());
    mlir::Value columnCount = valueMap.lookup(gemm.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(gemm.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(gemm,
                                         "repack-gemm-q4_1 ABI operand unmapped");

    llvm::StringRef opName = gemm.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one notch
    // i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!gemm.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          gemm, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *gemm.getIntegerCoreLmul();
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
    // The decode runs on the UNSIGNED weight lane (q4_1 is asymmetric, no
    // offset-binary bias); the activations stay i8.
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

    // The 16x1 repacked q4_1 GEMM block-format structural facts (I4 mirror, pinned
    // by the verifier): QK=32, block_q4_1x16 weight stride 320 (16 d + 16 m + 256
    // nibble bytes), block_q8_1x4 activation stride 144 (4 d + 4 s + 128 int8
    // quants), the weight nibble bytes at +64, the per-row MIN strip at +32, the
    // activation int8 quants at +16, the per-column activation scaled-sum at +8,
    // 16 weight rows / 4 activation columns per group, and the VLEN-derived e16m1
    // half width.
    int64_t qk = gemm.getQk();
    int64_t weightStride = gemm.getWeightBlockStride();
    int64_t activationStride = gemm.getActivationBlockStride();
    int64_t weightQuantOffset = gemm.getWeightQuantByteOffset();   // 64
    int64_t activationQuantOffset = gemm.getActivationQuantByteOffset(); // 16
    int64_t weightMinOffset = gemm.getWeightMinByteOffset();       // 32
    int64_t activationSumOffset = gemm.getActivationSumByteOffset(); // 8
    int64_t weightInterleave = gemm.getWeightInterleave();   // 16
    int64_t activationInterleave = gemm.getActivationInterleave(); // 4
    int64_t half = gemm.getHalfLanes();                      // 8 @128, 16 @256
    int64_t numHalves = weightInterleave / half;             // 2 @128, 1 @256
    // RVV1.0 (fractional chain) holds all 4 columns at once in one pass; RVV0.7.1
    // (whole-LMUL chain) doubles every rung and must fold ONE column per pass to
    // keep the per-pass live set under 32 vregs (the identical spill-avoidance
    // rationale the q4_0 GEMM documents at length).
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave;       // 1 @rvv07; 4 @rvv1.0
    int64_t nibbleBytes = qk / 2;                            // 16 nibble bytes
    // The activation high-half int8 quants start after the 64 low-half quants (4
    // columns x 16 lanes the low half consumes per nibble step). qs[16 + 64 + i*4 + c].
    int64_t activationHighRow = activationInterleave * nibbleBytes; // 64

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemm,
                                         "repack-gemm-q4_1 output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width.
    mlir::Value vl8 = sizeLit(half);

    // size_t nb = n / QK;
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

    // A typed UNSIGNED u8 contiguous sub-load of the raw repacked nibbles.
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // The UNSIGNED-nibble asymmetric decode: low = vand(b, 0x0F); high =
    // vsrl(b, 0x04); each reinterpret to i8 (value-identity for 0..15). NO
    // vsll/vsra sign-extend (the q4_1 bias lives in the separate MIN scale).
    std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    std::string reinterpretCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
    auto reinterpretToI8 = [&](mlir::Value u) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{u}, opName, role);
    };
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      mlir::Value lo = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vandCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value mask =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x0F")
                    .getResult();
            return {packed, mask, vl8};
          });
      return reinterpretToI8(lo);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      mlir::Value hi = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vsrlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value four =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x04")
                    .getResult();
            return {packed, four, vl8};
          });
      return reinterpretToI8(hi);
    };
    // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k].
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
    // vwmacc_vx widening multiply-accumulate: acc += scalar * vec (i8->i16).
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmaccCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
    };

    // ===== Outer activation-ROW-GROUP loop: for (y = 0; y < nr/4; ++y) =====
    auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    mlir::LogicalResult status = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();

      // const uint8_t *a = vy + y*nb*144;  (the q8_1x4 row group base).
      step("act_group_base");
      mlir::Value aGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, y, nb);
      mlir::Value aGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, aGroupBlocks, sizeLit(activationStride));
      mlir::Value aGroup = rewriter.create<emitc::AddOp>(
          loc, activationPtrType, activationBase, aGroupOff);

      // ===== Weight-COLUMN-GROUP loop: for (x = 0; x < nc/16; ++x) =====
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard cg(rewriter);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        mlir::Value x = colLoop.getInductionVar();

        // const uint8_t *b = vx + x*nb*320;  (the q4_1x16 column group base).
        step("weight_group_base");
        mlir::Value bGroupBlocks =
            rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
        mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, bGroupBlocks, sizeLit(weightStride));
        mlir::Value bGroup = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, weightBase, bGroupOff);

        // ===== Strip loop over the num_halves strips (2 of 8 @128, 1 of 16 @256).
        auto halfLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                      sizeLit(numHalves),
                                                      sizeLit(1),
                                                      /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard hg(rewriter);
          rewriter.setInsertionPointToStart(halfLoop.getBody());
          mlir::Value h = halfLoop.getInductionVar();
          step("half_row_offset");
          mlir::Value roff =
              rewriter.create<emitc::MulOp>(loc, sizeType, h, sizeLit(half));

          // Activation-column-PASS loop (compile-time, C++): the columns
          // [cLo, cLo+columnsPerPass) folded in this pass over the block loop.
          for (int64_t cLo = 0; cLo < activationInterleave;
               cLo += columnsPerPass) {
          int64_t cHi = cLo + columnsPerPass;
          // vfloat32m2_t sumf_{cLo..cHi} = vfmv_v_f(0.0f, half);  (per-pass f32 acc)
          std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
          llvm::SmallVector<mlir::Value> sumf(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            sumf[c] = emitOpaqueCallBuilt(
                rewriter, loc, f32m2Type, fmvCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  mlir::Value zero =
                      rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                          .getResult();
                  return {zero, vl8};
                });
          }
          llvm::SmallVector<mlir::Value> sumfVar(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            auto v = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, v, sumf[c]);
            sumfVar[c] = v;
          }

          // ===== Inner contraction-BLOCK loop: for (l = 0; l < nb; ++l) =====
          auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                         sizeLit(1),
                                                         /*bodyBuilder=*/nullptr);
          {
            mlir::OpBuilder::InsertionGuard bg(rewriter);
            rewriter.setInsertionPointToStart(blockLoop.getBody());
            mlir::Value l = blockLoop.getInductionVar();

            // const uint8_t *bl = b + l*320;   const uint8_t *al = a + l*144;
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

            // vint16m1_t sumi_{c}_{lo,hi} = vmv_v_x(0, half);
            std::string mvCallee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
            auto seedI16 = [&]() -> mlir::Value {
              return emitOpaqueCallBuilt(
                  rewriter, loc, i16m1Type, mvCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value zero =
                        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                            .getResult();
                    return {zero, vl8};
                  });
            };
            llvm::SmallVector<mlir::Value> sumiLoVar(activationInterleave),
                sumiHiVar(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c) {
              auto vlo = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i16m1Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, vlo, seedI16());
              sumiLoVar[c] = vlo;
            }
            for (int64_t c = cLo; c < cHi; ++c) {
              auto vhi = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i16m1Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, vhi, seedI16());
              sumiHiVar[c] = vhi;
            }

            // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
            auto nibLoop = rewriter.create<emitc::ForOp>(
                loc, sizeLit(0), sizeLit(nibbleBytes), sizeLit(1),
                /*bodyBuilder=*/nullptr);
            {
              mlir::OpBuilder::InsertionGuard ng(rewriter);
              rewriter.setInsertionPointToStart(nibLoop.getBody());
              mlir::Value i = nibLoop.getInductionVar();

              // b_packed = vle8_u8(&bl.qs[i*16 + roff], half);  byte = 64+i*16+roff.
              step("weight_nibble_addr");
              mlir::Value i16 = rewriter.create<emitc::MulOp>(
                  loc, sizeType, i, sizeLit(weightInterleave));
              mlir::Value qsOff = rewriter.create<emitc::AddOp>(
                  loc, sizeType, sizeLit(weightQuantOffset), i16);
              mlir::Value wByteOff =
                  rewriter.create<emitc::AddOp>(loc, sizeType, qsOff, roff);
              mlir::Value packed = loadNibbles(bl, wByteOff);
              mlir::Value bLo = decodeLo(packed);
              mlir::Value bHi = decodeHi(packed);

              // i*4 (the activation column-quant stride for the low/high halves).
              mlir::Value i4 = rewriter.create<emitc::MulOp>(
                  loc, sizeType, i, sizeLit(activationInterleave));

              for (int64_t c = cLo; c < cHi; ++c) {
                // sumi_c_lo = vwmacc_vx(sumi_c_lo, al.qs[16 + i*4+c], b_lo, half);
                step("act_quant_addr_lo");
                mlir::Value loIdx = rewriter.create<emitc::AddOp>(
                    loc, sizeType, i4, sizeLit(c));
                mlir::Value loOff = rewriter.create<emitc::AddOp>(
                    loc, sizeType, sizeLit(activationQuantOffset), loIdx);
                mlir::Value aLo = i8Read(al, loOff);
                mlir::Value curLo =
                    rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[c])
                        .getResult();
                rewriter.create<emitc::AssignOp>(loc, sumiLoVar[c],
                                                 vwmacc(curLo, aLo, bLo));

                // sumi_c_hi = vwmacc_vx(sumi_c_hi, al.qs[16+64+i*4+c], b_hi, half).
                step("act_quant_addr_hi");
                mlir::Value hiIdx = rewriter.create<emitc::AddOp>(
                    loc, sizeType, i4, sizeLit(c));
                mlir::Value hiBase = rewriter.create<emitc::AddOp>(
                    loc, sizeType, sizeLit(activationQuantOffset),
                    sizeLit(activationHighRow));
                mlir::Value hiOff = rewriter.create<emitc::AddOp>(
                    loc, sizeType, hiBase, hiIdx);
                mlir::Value aHi = i8Read(al, hiOff);
                mlir::Value curHi =
                    rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[c])
                        .getResult();
                rewriter.create<emitc::AssignOp>(loc, sumiHiVar[c],
                                                 vwmacc(curHi, aHi, bHi));
              }
            }

            // const vint32m2_t sumi_c = vwadd_vv(sumi_c_lo, sumi_c_hi, half).
            std::string vwaddCallee = ("__riscv_vwadd_vv_i32" + l32).str();
            llvm::SmallVector<mlir::Value> sumi32(activationInterleave);
            for (int64_t c = cLo; c < cHi; ++c) {
              mlir::Value lo =
                  rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[c])
                      .getResult();
              mlir::Value hi =
                  rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[c])
                      .getResult();
              sumi32[c] =
                  emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                                 mlir::ValueRange{lo, hi, vl8}, opName, role);
            }

            // vfloat16m1_t b_d = vle16(&bl.d[roff], half);  byte = roff*2.
            // vfloat16m1_t b_m = vle16(&bl.m[roff], half);  byte = 32 + roff*2.
            std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
            auto loadF16Strip = [&](int64_t baseByteOff) -> mlir::Value {
              step("weight_scale_addr");
              mlir::Value roff2 = rewriter.create<emitc::MulOp>(
                  loc, sizeType, roff, sizeLit(2));
              mlir::Value totalOff = roff2;
              if (baseByteOff != 0)
                totalOff = rewriter.create<emitc::AddOp>(
                    loc, sizeType, roff2, sizeLit(baseByteOff));
              mlir::Value dFull = rewriter.create<emitc::AddOp>(
                  loc, weightPtrType, bl, totalOff);
              mlir::Value dCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull)
                      .getResult();
              return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                                    mlir::ValueRange{dCast, vl8}, opName, role);
            };
            mlir::Value bD = loadF16Strip(0);
            mlir::Value bM = loadF16Strip(weightMinOffset);

            // Per-column fold ggml's q4_1 statement
            //   sumf_c += (d_x*d_y_c)*sumi_c + m_x*s_y_c:
            //   d_c = vfwmul_vf(b_d, *(_Float16*)&al.d[c]);  // (d_x * d_y_c)
            //   sumf_c = vfmacc_vv(sumf_c, vfcvt(sumi_c), d_c);
            //   m_c = vfwmul_vf(b_m, *(_Float16*)&al.s[c]);  // (m_x * s_y_c)
            //   sumf_c = vfadd_vv(sumf_c, m_c);
            std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
            std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32,
                                                         "f32");
            std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
            std::string vfaddCallee = ("__riscv_vfadd_vv_f32" + l32).str();
            llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
            mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
            for (int64_t c = cLo; c < cHi; ++c) {
              // d_y_c = *(const _Float16 *)&al.d[c]  (al + c*2).
              mlir::Value aD = emitOpaqueCallBuilt(
                  rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value aDOff = rewriter.create<emitc::AddOp>(
                        loc, activationPtrType, al, sizeLit(c * 2));
                    mlir::Value aDCast =
                        rewriter.create<emitc::CastOp>(loc, f16PtrType, aDOff)
                            .getResult();
                    return {aDCast};
                  },
                  llvm::StringRef("act_scale_scalar"));
              // Scale term (d_x*d_y_c)*sumi_c.
              mlir::Value dC =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                                 mlir::ValueRange{bD, aD, vl8}, opName, role);
              mlir::Value sumiF =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                 mlir::ValueRange{sumi32[c], vl8}, opName, role);
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                      .getResult();
              mlir::Value scaled =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                                 mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                                 role);
              // s_y_c = *(const _Float16 *)&al.s[c]  (al + 8 + c*2).
              mlir::Value aS = emitOpaqueCallBuilt(
                  rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value aSOff = rewriter.create<emitc::AddOp>(
                        loc, activationPtrType, al,
                        sizeLit(activationSumOffset + c * 2));
                    mlir::Value aSCast =
                        rewriter.create<emitc::CastOp>(loc, f16PtrType, aSOff)
                            .getResult();
                    return {aSCast};
                  },
                  llvm::StringRef("act_sum_scalar"));
              // MIN term m_x*s_y_c, added LANE-WISE.
              mlir::Value mC =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                                 mlir::ValueRange{bM, aS, vl8}, opName, role);
              mlir::Value nextF =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfaddCallee,
                                 mlir::ValueRange{scaled, mC, vl8}, opName,
                                 role);
              rewriter.create<emitc::AssignOp>(loc, sumfVar[c], nextF);
            }
          }

          // vse32(s + (y*4 + c)*bs + x*16 + roff, sumf_c, half);  the 4x8 store.
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
            mlir::Value totalOff =
                rewriter.create<emitc::AddOp>(loc, sizeType, colOff, roff);
            mlir::Value dst = rewriter.create<emitc::AddOp>(
                loc, floatPtrType, output, totalOff);
            mlir::Value sumfVal =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                    .getResult();
            emitOpaqueCallVoid(rewriter, loc, vseCallee,
                               mlir::ValueRange{dst, sumfVal, vl8}, opName,
                               role);
          }
          } // end activation-column-PASS loop (cLo)
        }
      }
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    // The op result is the typed i32m1 token; the GEMM writes through *s so the
    // token has no consumer. Seed it with a zero i32m1 lane-0.
    std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
    mlir::Value zeroLane =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
    mlir::Value resultTok =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                         seedCallee,
                                         mlir::ValueRange{zeroLane, sizeLit(1)})
            .getResult(0);
    valueMap[gemm.getResult()] = resultTok;
    return mlir::success();
  }

// q4_K x q8_K 16x1-REPACKED PREFILL GEMM emitter (stage-1b-iii). The dominant-quant prefill
// e2e-win path: the q4_K GEVM's oracle-verified 8-sub-block 6-bit scale/min lane-wise unpack +
// dual d/dmin fold (emitRepackGemvQ4KQ8K), with the WEIGHT-side unpack done ONCE per 16-weight
// group and REUSED across the M (=4) interleaved activation columns of block_q8_Kx4 -- the
// amortization the single-column GEVM lacks. Activation interleave (pinned from ggml
// ggml_gemm_q4_K_16x1_q8_K_generic + ggml_quantize_mat_q8_K_4x1, repack.cpp:2442/:90): qs@16 are
// 4-column-interleaved (element e of column c at qs[e*4 + c]); bsums@1040 are group16-major /
// column-minor (group g16 col c at bsums[g16*4 + c]); d[4]@0 are 4 fp32 (NOT fp16) scalars.
// NUMERIC STATUS: structurally complete AND oracle-verified — vs an INDEPENDENT scalar
// dequant-matmul reference (M=4) on the RVV1.0 VLEN128 board (bounded-norm PASS, WORST_NORM
// ~7e-07 over 8 shapes; controls NOMIN 3.5e5x / PERM 4e6x / ROWROT 7.6e6x). NOT byte-exact vs
// ggml_gemm_q4_K_16x1_q8_K (IEEE-legal float reassociation only, not a decode error).

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
