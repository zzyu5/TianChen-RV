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

// VariantToEmitCFunc block-quant linear (block-dot / GEMM) emit methods:
// q4_0/q8_0, q4_1/q5_0/q5_1, q1_0 + the q4_0 gemm-tile / gemm tilings. Split out
// of RVVToEmitC.cpp as a pure code move; the emitted C is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitQ4_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: q4_0 offset_binary_nibble / half-block / LeftAssoc instance of
    // the descriptor-driven emitFlatBlockDot. Resolve ABI/provenance, project
    // the formula-produced final flat_* plan plus raw typed geometry, and emit
    // the shared body without a kind/format decision replay.
    weftrvv::GgmlBlockDotQ40Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<weftrvv::GgmlBlockDotQ40Q80Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot ABI operand unmapped");

    std::optional<FlatBlockDotDescriptor> descriptor =
        readFinalFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-plain");
    std::optional<BlockDotFacts> facts = readFinalBlockDotFacts(blockDot);
    if (!facts)
      return rewriter.notifyMatchFailure(
          blockDot, "block-dot reached emission without a complete final "
                    "schedule");
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getWEFTEmitCLowerableSourceOpName(),
                            blockDot.getWEFTEmitCLowerableSourceRole(), *facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_0Q8_0GemmTile(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlGemmTileQ40Q80Op tile;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto t = llvm::dyn_cast<weftrvv::GgmlGemmTileQ40Q80Op>(op))
        tile = t;
    }
    if (!tile)
      return rewriter.notifyMatchFailure(scope, "gemm-tile body missing the op");

    mlir::Value weightBase = valueMap.lookup(tile.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(tile.getActivationBase());
    mlir::Value columnStride = valueMap.lookup(tile.getActivationColumnStride());
    mlir::Value output = valueMap.lookup(tile.getOutput());
    if (!weightBase || !activationBase || !columnStride || !output)
      return rewriter.notifyMatchFailure(tile, "gemm-tile ABI operand unmapped");

    llvm::StringRef opName = tile.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = tile.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = tile.getQk();
    int64_t weightStride = tile.getWeightBlockStride();
    int64_t activationStride = tile.getActivationBlockStride();
    int64_t quantOffset = tile.getQuantByteOffset();
    int64_t highOffset = tile.getActivationHighByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block
    int64_t cols = tile.getActivationCols();

    // The weight decode anchors at the m1 whole-half-block form (one
    // vsetvl_e8m1(16) covers the 16 nibble bytes at VLEN >= 128); the product
    // widens i8m1 -> i16m2. These are the *how* (vector grouping), never the
    // *what*: the dot product is byte-exact (vwredsum sums the same integer set).
    // [A-line stage-3: structural constant, NOT a knob] the m1/m2 shape is FIXED by
    // this GEMM-tile body (hardcoded vint8m1_t/vint16m2_t types + the "m1"/"m2"
    // intrinsic-name suffixes below all pin the same shape); there is no IR width to
    // read. [K-10]: do not fake a knob -- the debake is a no-op here.
    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf[M];  (the M INDEPENDENT fp32 column accumulators, an emitc
    // array). Zero each lane: for (size_t j = 0; j < M; ++j) sumf[j] = 0.0f;
    // -- emitted as M explicit assigns so the init is a simple structured node
    // sequence (M is a small bounded compile-time tile).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    mlir::Type sumfArrayType = emitc::ArrayType::get({cols}, floatType);
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, sumfArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sumfArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumfVar.getResult());
    auto sumfElem = [&](mlir::Value j) -> mlir::Value {
      return rewriter
          .create<emitc::SubscriptOp>(loc, sumfArray, mlir::ValueRange{j})
          .getResult();
    };
    for (int64_t j = 0; j < cols; ++j) {
      mlir::Value jIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(j));
      rewriter.create<emitc::AssignOp>(
          loc, sumfElem(jIdx),
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
    }

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // Per-block weight address arithmetic: const uint8_t *xb = vx + ib*18.
    auto blockBaseValue = [&](mlir::Value idx, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The scalar fp16->fp32 read (the ONE sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    // A typed i8m1 chunk load: __riscv_vle8_v_i8m1(ptr + fixed, vl).
    std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    auto loadChunk = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                         mlir::Value vl) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8CoreType, loadCallee,
                            mlir::ValueRange{cast, vl}, opName, role);
    };

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the AoS weight-block
    // loop. The weight decode is hoisted to the TOP of this body (once per
    // block); the inner M-column loop reuses it.
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    mlir::LogicalResult blockStatus = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // Shared weight block: address + fp16 scale + HOISTED decode.
      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "block_base_x");
      mlir::Value dX = fp16Read(xb);

      // size_t vl = __riscv_vsetvl_e8m1(16);  (m1 whole-half-block, VLEN>=128).
      std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
      mlir::Value vl = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, setvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(halfBlock)};
          });

      // vint8m1_t w = vle8(xb + 2, vl);  then the offset-binary decode into
      // v0/v1 -- the SHARED decoded nibble lanes reused across all M columns.
      mlir::Value w = loadChunk(xb, weightPtrType, quantOffset, vl);
      std::pair<mlir::Value, mlir::Value> decoded = emitOffsetBinaryDecodeValue(
          rewriter, loc, w, vl, i8CoreType, "i8", coreLmul, opName, role);

      // The inner M-column loop. Each column j: address (vy + j*by + ib*34),
      // its fp16 scale, its two q8 halves, the product against the HOISTED
      // v0/v1, the per-column reduce, and the ascending-block-order fp32 fold
      // into sumf[j]. M independent accumulators -> M byte-exact vec_dot results.
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                   sizeLit(cols), sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::LogicalResult colStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard colGuard(rewriter);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        mlir::Value j = colLoop.getInductionVar();

        // const uint8_t *yb = vy + j*by + ib*34;
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "column_base_y"));
        mlir::Value colOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, j, columnStride);
        mlir::Value ybCol = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, activationBase, colOff);
        mlir::Value yb = blockBaseValue(ib, ybCol, activationPtrType,
                                        activationStride, "block_base_y");
        mlir::Value dY = fp16Read(yb);

        // vint8m1_t y0 = vle8(yb + 2, vl);  vint8m1_t y1 = vle8(yb + 2 + 16, vl);
        mlir::Value y0 = loadChunk(yb, activationPtrType, quantOffset, vl);
        mlir::Value y1 =
            loadChunk(yb, activationPtrType, quantOffset + highOffset, vl);

        // The product half against the HOISTED decoded weight lanes (byte-
        // identical nodes to the per-row block dot's vwmul/vwmacc).
        mlir::FailureOr<mlir::Value> product =
            emitOffsetBinaryProductFromDecodedValue(
                rewriter, loc, decoded.first, decoded.second, y0, y1, vl,
                i16WideType, 16, wideLmul, "i16", opName, role);
        if (mlir::failed(product)) {
          colStatus = mlir::failure();
        } else {
          // Per-column reduce: seed lane0 = 0 (the m1 strip runs once at
          // VLEN >= 128), vwredsum, extract scalar sumi.
          std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          mlir::Value seed = emitOpaqueCallBuilt(
              rewriter, loc, i32m1Type, seedCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value sumiSeed =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {sumiSeed, sizeLit(1)};
              });
          std::string reduceCallee =
              ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
          mlir::Value red =
              emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                             mlir::ValueRange{*product, seed, vl}, opName, role);
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          mlir::Value sumi =
              emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                             mlir::ValueRange{red}, opName, role);

          // sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  -- ggml's exact
          // left-associative order, grouped into ONE emitc.expression so
          // mlir-translate renders it as a SINGLE C statement the compiler
          // fuses into the SAME FMA ggml does under -ffp-contract=on/default
          // (byte-exact across all four modes). The emitc.load of sumf[j] stays
          // OUTSIDE the expression (load lacks the CExpression trait). Each
          // column accumulates ib ascending into its OWN sumf[j], so the fp32
          // non-associativity boundary is per-column-identical to vec_dot.
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "fp32_accumulate"));
          mlir::Value sumfElemLval = sumfElem(j);
          mlir::Value sumfCur =
              rewriter.create<emitc::LoadOp>(loc, floatType, sumfElemLval)
                  .getResult();
          auto accumExpr = rewriter.create<emitc::ExpressionOp>(
              loc, floatType, /*do_not_inline=*/false);
          {
            mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
            mlir::Block *exprBlock =
                rewriter.createBlock(&accumExpr.getRegion());
            rewriter.setInsertionPointToStart(exprBlock);
            mlir::Value sumiFloat =
                rewriter.create<emitc::CastOp>(loc, floatType, sumi)
                    .getResult();
            mlir::Value timesDx =
                rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
            mlir::Value blockTerm =
                rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
            mlir::Value sumfNext =
                rewriter.create<emitc::AddOp>(loc, floatType, sumfCur,
                                              blockTerm);
            rewriter.create<emitc::YieldOp>(loc, sumfNext);
          }
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumf", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumfElemLval,
                                           accumExpr.getResult());
        }
      }
      if (mlir::failed(colStatus))
        blockStatus = mlir::failure();
    }
    if (mlir::failed(blockStatus))
      return mlir::failure();

    // for (j) s[j] = sumf[j];  -- the M-output store through the float * pointer
    // (M explicit structured assigns; s[0..M-1] contiguous, the bs ABI stride
    // is G2).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(tile, "gemm-tile output not a pointer");
    mlir::Value lastStored;
    for (int64_t j = 0; j < cols; ++j) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_s"));
      mlir::Value jIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(j));
      emitc::SubscriptOp outSubscript =
          rewriter.create<emitc::SubscriptOp>(loc, outPointer, jIdx);
      mlir::Value sumfVal =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfElem(jIdx))
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfVal);
      lastStored = sumfVal;
    }

    valueMap[tile.getResult()] = lastStored;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_0Q8_0Gemm(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlGemmQ40Q80Op gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<weftrvv::GgmlGemmQ40Q80Op>(op))
        gemm = g;
    }
    if (!gemm)
      return rewriter.notifyMatchFailure(scope, "gemm body missing the op");

    mlir::Value weightBase = valueMap.lookup(gemm.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemm.getActivationBase());
    mlir::Value columnStride = valueMap.lookup(gemm.getActivationColumnStride());
    mlir::Value output = valueMap.lookup(gemm.getOutput());
    mlir::Value rowCount = valueMap.lookup(gemm.getRowCount());
    mlir::Value columnCount = valueMap.lookup(gemm.getColumnCount());
    mlir::Value weightRowStride = valueMap.lookup(gemm.getWeightRowStride());
    mlir::Value outputRowStride = valueMap.lookup(gemm.getOutputRowStride());
    if (!weightBase || !activationBase || !columnStride || !output ||
        !rowCount || !columnCount || !weightRowStride || !outputRowStride)
      return rewriter.notifyMatchFailure(gemm, "gemm ABI operand unmapped");

    llvm::StringRef opName = gemm.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = gemm.getQk();
    int64_t weightStride = gemm.getWeightBlockStride();
    int64_t activationStride = gemm.getActivationBlockStride();
    int64_t quantOffset = gemm.getQuantByteOffset();
    int64_t highOffset = gemm.getActivationHighByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block
    // M is a final formula result. Emission consumes it mechanically and cannot
    // recover an analytic or measured choice from absence.
    if (!gemm.getActivationCols())
      return rewriter.notifyMatchFailure(
          gemm, "GEMM reached emission without final activation_cols");
    int64_t cols = *gemm.getActivationCols();

    // The weight decode anchors at the m1 whole-half-block form (one
    // vsetvl_e8m1(16) covers the 16 nibble bytes at VLEN >= 128); the product
    // widens i8m1 -> i16m2. The *how* (vector grouping), never the *what*: the
    // dot product is byte-exact (vwredsum sums the same integer set).
    // [A-line stage-3: structural constant, NOT a knob] the m1/m2 shape is FIXED by
    // this GEMM body (hardcoded vint8m1_t/vint16m2_t types + intrinsic-name suffixes
    // below); no IR width to read. [K-10]: do not fake a knob (debake is a no-op).
    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The scalar fp16->fp32 read (the ONE sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    // A typed i8m1 chunk load: __riscv_vle8_v_i8m1(ptr + fixed, vl).
    std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    auto loadChunk = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                         mlir::Value vl) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8CoreType, loadCallee,
                            mlir::ValueRange{cast, vl}, opName, role);
    };

    // size_t nb = n / QK;  (the contraction block count, shared across rows/cols)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemm, "gemm output not a pointer");

    // ===== Outer weight-ROW loop: for (size_t ir = 0; ir < nr; ++ir) =====
    auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), rowCount,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    mlir::LogicalResult rowStatus = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value ir = rowLoop.getInductionVar();

      // const uint8_t *xr = vx + ir*bx;   float *sr = s + ir*bs;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "weight_row_base"));
      mlir::Value rowWeightOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ir, weightRowStride);
      mlir::Value xr = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                     weightBase, rowWeightOff);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "output_row_base"));
      mlir::Value rowOutOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ir, outputRowStride);
      mlir::Value sr = rewriter.create<emitc::AddOp>(loc, floatPtrType, output,
                                                     rowOutOff);

      auto srPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(sr);

      // emitStrip(cb, colBound): one M-wide column strip starting at column cb,
      // processing `colBound` columns (colBound is the inner-loop trip count).
      // The FULL strips pass the COMPILE-TIME CONSTANT M (sizeLit(cols)) so the
      // inner column loop is a constant-trip loop the C compiler fully unrolls
      // -- recovering G1's tile shape (the per-column vwredsum/vmv_x_s/fp32-fold
      // chains overlap across the unrolled columns; a runtime bound serializes
      // them and ships a regression). The ONE tail strip passes the runtime
      // remainder (nc % M). The math is identical for either bound, so every
      // output stays byte-exact vs per-(row,col) vec_dot; only the inner
      // trip-count SHAPE differs (the same "full groups + tail" pattern the
      // block dot's multi_block_factor uses).
      auto emitStrip = [&](mlir::Value cb,
                           mlir::Value colBound) -> mlir::LogicalResult {
        // const uint8_t *yb0 = vy + cb*by;  -- the first activation column of
        // this strip; column j of the strip is yb0 + j*by.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "strip_base_y"));
        mlir::Value stripColOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, cb, columnStride);
        mlir::Value yb0 = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, activationBase, stripColOff);

        // float sumf[M]; for (j < colBound) sumf[j] = 0.0f;  -- the M-wide fp32
        // accumulator array (M bounded). The init is over colBound (a partial
        // tail strip leaves the unused lanes untouched).
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumf", opName, role));
        mlir::Type sumfArrayType = emitc::ArrayType::get({cols}, floatType);
        auto sumfVar = rewriter.create<emitc::VariableOp>(
            loc, sumfArrayType, emitc::OpaqueAttr::get(ctx, ""));
        auto sumfArray =
            llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumfVar.getResult());
        auto sumfElem = [&](mlir::Value j) -> mlir::Value {
          return rewriter
              .create<emitc::SubscriptOp>(loc, sumfArray, mlir::ValueRange{j})
              .getResult();
        };
        {
          auto zeroLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                        colBound, sizeLit(1),
                                                        /*bodyBuilder=*/nullptr);
          mlir::OpBuilder::InsertionGuard zg(rewriter);
          rewriter.setInsertionPointToStart(zeroLoop.getBody());
          rewriter.create<emitc::AssignOp>(
              loc, sumfElem(zeroLoop.getInductionVar()),
              rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
        }

        // ===== The weight-decode-reuse block loop (G1's tile body) =====
        // for (size_t ib = 0; ib < nb; ++ib) { decode weight ONCE; for j<colBound }
        auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        mlir::LogicalResult blockStatus = mlir::success();
        {
          mlir::OpBuilder::InsertionGuard blockGuard(rewriter);
          rewriter.setInsertionPointToStart(blockLoop.getBody());
          mlir::Value ib = blockLoop.getInductionVar();

          // Shared weight block: address (xr + ib*18) + fp16 scale + HOISTED
          // decode into v0/v1, computed ONCE and reused across the strip cols.
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "block_base_x"));
          mlir::Value xbOff =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib,
                                            sizeLit(weightStride));
          mlir::Value xb = rewriter.create<emitc::AddOp>(loc, weightPtrType, xr,
                                                         xbOff);
          mlir::Value dX = fp16Read(xb);

          // size_t vl = __riscv_vsetvl_e8m1(16);  (m1 whole-half-block, VLEN>=128)
          std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
          mlir::Value vl = emitOpaqueCallBuilt(
              rewriter, loc, sizeType, setvlCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {sizeLit(halfBlock)};
              });

          // vint8m1_t w = vle8(xb + 2, vl);  then offset-binary decode -> v0/v1
          // (the SHARED decoded nibble lanes reused across all strip columns).
          mlir::Value w = loadChunk(xb, weightPtrType, quantOffset, vl);
          std::pair<mlir::Value, mlir::Value> decoded =
              emitOffsetBinaryDecodeValue(rewriter, loc, w, vl, i8CoreType,
                                          "i8", coreLmul, opName, role);

          // ===== The inner COLUMN loop (reuses the hoisted v0/v1) =====
          // for (size_t j = 0; j < colBound; ++j) { ... }  -- colBound is the
          // constant M for full strips (unrollable) or the runtime tail.
          auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), colBound,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
          mlir::LogicalResult colStatus = mlir::success();
          {
            mlir::OpBuilder::InsertionGuard colGuard(rewriter);
            rewriter.setInsertionPointToStart(colLoop.getBody());
            mlir::Value j = colLoop.getInductionVar();

            // const uint8_t *yb = yb0 + j*by + ib*34;
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "column_base_y"));
            mlir::Value colOff =
                rewriter.create<emitc::MulOp>(loc, sizeType, j, columnStride);
            mlir::Value ybCol = rewriter.create<emitc::AddOp>(
                loc, activationPtrType, yb0, colOff);
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "block_base_y"));
            mlir::Value ybBlockOff =
                rewriter.create<emitc::MulOp>(loc, sizeType, ib,
                                              sizeLit(activationStride));
            mlir::Value yb = rewriter.create<emitc::AddOp>(
                loc, activationPtrType, ybCol, ybBlockOff);
            mlir::Value dY = fp16Read(yb);

            // vint8m1_t y0 = vle8(yb+2, vl);  y1 = vle8(yb+2+16, vl);
            mlir::Value y0 = loadChunk(yb, activationPtrType, quantOffset, vl);
            mlir::Value y1 = loadChunk(yb, activationPtrType,
                                       quantOffset + highOffset, vl);

            // The product half against the HOISTED decoded weight lanes (byte-
            // identical nodes to the per-row block dot's vwmul/vwmacc).
            mlir::FailureOr<mlir::Value> product =
                emitOffsetBinaryProductFromDecodedValue(
                    rewriter, loc, decoded.first, decoded.second, y0, y1, vl,
                    i16WideType, 16, wideLmul, "i16", opName, role);
            if (mlir::failed(product)) {
              colStatus = mlir::failure();
            } else {
              // Per-column reduce: seed lane0 = 0 (the m1 strip runs once at
              // VLEN >= 128), vwredsum, extract scalar sumi.
              std::string seedCallee =
                  riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
              mlir::Value seed = emitOpaqueCallBuilt(
                  rewriter, loc, i32m1Type, seedCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    mlir::Value sumiSeed =
                        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                            .getResult();
                    return {sumiSeed, sizeLit(1)};
                  });
              std::string reduceCallee =
                  ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
              mlir::Value red =
                  emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                                 mlir::ValueRange{*product, seed, vl}, opName,
                                 role);
              std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
              mlir::Value sumi =
                  emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                                 mlir::ValueRange{red}, opName, role);

              // sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  -- ggml's exact
              // left-associative order, grouped into ONE emitc.expression so
              // mlir-translate renders it as a SINGLE C statement the compiler
              // fuses into the SAME FMA ggml does under -ffp-contract=on/default
              // (byte-exact across all four modes). Each column accumulates ib
              // ascending into its OWN sumf[j], so the fp32 non-associativity
              // boundary is per-column-identical to vec_dot.
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, "fp32_accumulate"));
              mlir::Value sumfElemLval = sumfElem(j);
              mlir::Value sumfCur =
                  rewriter.create<emitc::LoadOp>(loc, floatType, sumfElemLval)
                      .getResult();
              auto accumExpr = rewriter.create<emitc::ExpressionOp>(
                  loc, floatType, /*do_not_inline=*/false);
              {
                mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
                mlir::Block *exprBlock =
                    rewriter.createBlock(&accumExpr.getRegion());
                rewriter.setInsertionPointToStart(exprBlock);
                mlir::Value sumiFloat =
                    rewriter.create<emitc::CastOp>(loc, floatType, sumi)
                        .getResult();
                mlir::Value timesDx =
                    rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
                mlir::Value blockTerm =
                    rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
                mlir::Value sumfNext = rewriter.create<emitc::AddOp>(
                    loc, floatType, sumfCur, blockTerm);
                rewriter.create<emitc::YieldOp>(loc, sumfNext);
              }
              rewriter.create<emitc::VerbatimOp>(
                  loc, assignComment("sumf", opName, role));
              rewriter.create<emitc::AssignOp>(loc, sumfElemLval,
                                               accumExpr.getResult());
            }
          }
          if (mlir::failed(colStatus))
            blockStatus = mlir::failure();
        }
        if (mlir::failed(blockStatus))
          return mlir::failure();

        // for (size_t j = 0; j < colBound; ++j) sr[cb + j] = sumf[j];  -- the
        // strip output store through the row pointer.
        auto storeLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), colBound,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        mlir::OpBuilder::InsertionGuard sg(rewriter);
        rewriter.setInsertionPointToStart(storeLoop.getBody());
        mlir::Value j = storeLoop.getInductionVar();
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "store_s"));
        mlir::Value outIdx =
            rewriter.create<emitc::AddOp>(loc, sizeType, cb, j);
        emitc::SubscriptOp outSubscript =
            rewriter.create<emitc::SubscriptOp>(loc, srPtr, outIdx);
        mlir::Value sumfVal =
            rewriter.create<emitc::LoadOp>(loc, floatType, sumfElem(j))
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(),
                                         sumfVal);
        return mlir::success();
      };

      // size_t ncFull = (nc / M) * M;  -- the full-strip span (a multiple of M).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "full_strip_span"));
      mlir::Value ncGroups =
          rewriter.create<emitc::DivOp>(loc, sizeType, columnCount,
                                        sizeLit(cols));
      mlir::Value ncFull =
          rewriter.create<emitc::MulOp>(loc, sizeType, ncGroups, sizeLit(cols));

      // ===== FULL column-strip loop: for (cb = 0; cb < ncFull; cb += M) =====
      // The inner column loop trip count is the COMPILE-TIME CONSTANT M, so the
      // C compiler fully unrolls it (recovering G1's overlapping reductions).
      auto stripLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncFull,
                                                     sizeLit(cols),
                                                     /*bodyBuilder=*/nullptr);
      mlir::LogicalResult stripStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard stripGuard(rewriter);
        rewriter.setInsertionPointToStart(stripLoop.getBody());
        stripStatus = emitStrip(stripLoop.getInductionVar(), sizeLit(cols));
      }
      if (mlir::failed(stripStatus))
        rowStatus = mlir::failure();

      // ===== ONE tail strip: if (ncFull < nc) emitStrip(ncFull, nc-ncFull) =====
      // The final nc % M columns (0 when nc is a multiple of M). The tail's
      // inner column loop is the runtime remainder; it runs at most once per
      // row, so the (un-unrolled) tail does not dominate the cost.
      if (mlir::succeeded(rowStatus)) {
        mlir::Value tailCount =
            rewriter.create<emitc::SubOp>(loc, sizeType, columnCount, ncFull);
        mlir::Value hasTail =
            rewriter
                .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, ncFull,
                                      columnCount)
                .getResult();
        auto tailIf = rewriter.create<emitc::IfOp>(loc, hasTail,
                                                   /*addThenBlock=*/true,
                                                   /*addElseBlock=*/false);
        {
          mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
          rewriter.setInsertionPointToStart(&tailIf.getThenRegion().front());
          mlir::LogicalResult tailStatus = emitStrip(ncFull, tailCount);
          rewriter.create<emitc::YieldOp>(loc);
          if (mlir::failed(tailStatus))
            rowStatus = mlir::failure();
        }
      }
    }
    if (mlir::failed(rowStatus))
      return mlir::failure();

    // The op result is the typed i32m1 token; the GEMM writes through *s so the
    // token has no consumer. Seed it with a zero i32m1 lane-0 so the valueMap
    // entry is well-formed (mirrors the per-row block dot's unused result).
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

// The shared q4_0 16x1-REPACKED GEMM per-block LANE-WISE integer CORE leaf.
// Factored VERBATIM out of the now-RETIRED emitRepackGemmQ4_0Q8_0 monolith's
// block-loop body so the SAME node sequence (seed per-column i16 lo/hi -> nibble-step
// vwmacc loop over ONE strip at the runtime roff and the [cLo,cHi) interleaved
// activation columns -> per-column lo/hi vwadd combine) is driven by the first-class
// GEMM integer-core brick inside the typed weft_rvv.typed_repack_gemm_loop_body region
// (emitTypedRepackGemmLoopBody) -- the SOLE caller now the monolith is retired,
// byte-identity by construction preserved. Given the per-block bases bl/al (already
// advanced by
// block_index*stride) and the runtime strip row offset roff it returns the
// per-column i32 `sumi` values (indexed by absolute column, a vector of size
// activationInterleave filled at [cLo,cHi)); the per-column dual-fp16 scale fold
// that consumes them is the caller's.
llvm::SmallVector<mlir::Value>
VariantToEmitCFunc::emitRepackGemmQ4LaneWiseIntegerCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const RepackGemmQ4IntegerCoreContext &cx, mlir::Value bl, mlir::Value al,
    mlir::Value roff, int64_t cLo, int64_t cHi) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::StringRef opName = cx.opName;
  llvm::StringRef role = cx.role;
  llvm::StringRef l8 = cx.l8;
  llvm::StringRef l16 = cx.l16;
  llvm::StringRef l32 = cx.l32;
  mlir::Type sizeType = cx.sizeType;
  mlir::Value vl8 = cx.vl8;
  int64_t nibbleBytes = cx.nibbleBytes;
  int64_t weightInterleave = cx.weightInterleave;
  int64_t weightQuantOffset = cx.weightQuantOffset;
  int64_t activationQuantOffset = cx.activationQuantOffset;
  int64_t activationInterleave = cx.activationInterleave;
  int64_t activationHighRow = cx.activationHighRow;

  bool unsignedNibble = cx.unsignedNibble;
  bool hasQh = cx.hasQh;                       // q5_0 5th-bit decode (GEMM)
  bool fullI8 = cx.fullI8;                      // q8_0 full-int8 decode (GEMM)
  int64_t weightQhByteOffset = cx.weightQhByteOffset;
  int64_t offsetBias = cx.offsetBias;
  mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
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
  mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
  mlir::Type i8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
  mlir::Type u8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
  mlir::Type u16PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
  mlir::Type weightPtrType = bl.getType();
  mlir::Type activationPtrType = al.getType();

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto step = [&](llvm::StringRef s) {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
  };

  // q4_0: a typed i8 sub-load __riscv_vle8_v_i8<l8> (the repacked nibbles carry the
  // ^0x88 offset-binary bias). q4_1 (unsignedNibble): the asymmetric weight stores
  // RAW nibbles, so it loads unsigned __riscv_vle8_v_u8.
  std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
  std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
  auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
    mlir::Value full =
        rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
    if (unsignedNibble) {
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    }
    mlir::Value cast =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
    return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                          mlir::ValueRange{cast, vl8}, opName, role);
  };
  // q4_0 offset-binary decode = plain sign-extension: b_lo = vsra(vsll(b,4),4);
  // b_hi = vsra(b,4). q4_1 UNSIGNED decode = the RAW-nibble peel: b_lo =
  // vreinterpret_i8(vand(b,0x0F)); b_hi = vreinterpret_i8(vsrl(b,4)) (value-identity
  // for 0..15 -- the q4_1 bias lives in the separate MIN scale, NO sign-extend).
  std::string sllCallee = ("__riscv_vsll_vx_i8" + l8).str();
  std::string sraCallee = ("__riscv_vsra_vx_i8" + l8).str();
  std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
  std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
  std::string reinterpretCallee =
      ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
  mlir::Value four = sizeLit(4);
  auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
    if (unsignedNibble) {
      mlir::Value lo = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vandCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value mask =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x0F")
                    .getResult();
            return {packed, mask, vl8};
          });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{lo}, opName, role);
    }
    mlir::Value shl =
        emitOpaqueCall(rewriter, loc, i8mf2Type, sllCallee,
                       mlir::ValueRange{packed, four, vl8}, opName, role);
    return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                          mlir::ValueRange{shl, four, vl8}, opName, role);
  };
  auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
    if (unsignedNibble) {
      mlir::Value hi = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vsrlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sh =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x04")
                    .getResult();
            return {packed, sh, vl8};
          });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{hi}, opName, role);
    }
    return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                          mlir::ValueRange{packed, four, vl8}, opName, role);
  };

  // ===== q5_0 5th-bit (qh) decode leaf (hasQh), GEMM RUNTIME-strip form ========
  // The five-bit weight is `A = nibble | (qh_bit << 4)` in [0,31], reinterpreted
  // u8->i8, then `-offsetBias` (16). The strip's qh bit is selected by the RUNTIME
  // strip_row_offset `roff` (vid + roff), UNLIKE the GEVM compile-time h*half.
  std::string orCallee = ("__riscv_vor_vv_u8" + l8).str();
  std::string subCallee = ("__riscv_vsub_vx_i8" + l8).str();
  mlir::Value biasLit = sizeLit(offsetBias);
  auto nibbleLoU8 = [&](mlir::Value packed) -> mlir::Value {
    return emitOpaqueCallBuilt(
        rewriter, loc, u8mf2Type, vandCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {packed, sizeLit(15), vl8};
        });
  };
  auto nibbleHiU8 = [&](mlir::Value packed) -> mlir::Value {
    return emitOpaqueCall(rewriter, loc, u8mf2Type, vsrlCallee,
                          mlir::ValueRange{packed, four, vl8}, opName, role);
  };
  llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
  auto qhMaskScalar = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
    mlir::Value full =
        rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
    mlir::Value cast =
        rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
    return emitOpaqueCall(rewriter, loc, i32Type, u16ReadCallee,
                          mlir::ValueRange{cast}, opName, role,
                          llvm::StringRef("qh_mask_scalar"));
  };
  std::string vidCallee = ("__riscv_vid_v_u16" + l16).str();
  std::string vmvU16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "u16");
  std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
  std::string vsrlVvCallee = ("__riscv_vsrl_vv_u16" + l16).str();
  std::string vandU16Callee = ("__riscv_vand_vx_u16" + l16).str();
  std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
  std::string vncvtCallee = ("__riscv_vncvt_x_x_w_u8" + l8).str();
  // laneShiftVal is the RUNTIME strip_row_offset (index) -- vid + roff selects bit
  // (l + roff) for lane l of the strip at roff.
  auto expandQhBit = [&](mlir::Value maskScalar,
                         mlir::Value laneShiftVal) -> mlir::Value {
    mlir::Value splat =
        emitOpaqueCall(rewriter, loc, u16m1Type, vmvU16Callee,
                       mlir::ValueRange{maskScalar, vl8}, opName, role);
    mlir::Value vid = emitOpaqueCall(rewriter, loc, u16m1Type, vidCallee,
                                     mlir::ValueRange{vl8}, opName, role);
    vid = emitOpaqueCall(rewriter, loc, u16m1Type, vaddU16Callee,
                         mlir::ValueRange{vid, laneShiftVal, vl8}, opName, role);
    mlir::Value shifted =
        emitOpaqueCall(rewriter, loc, u16m1Type, vsrlVvCallee,
                       mlir::ValueRange{splat, vid, vl8}, opName, role);
    mlir::Value bit = emitOpaqueCallBuilt(
        rewriter, loc, u16m1Type, vandU16Callee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {shifted, sizeLit(1), vl8};
        });
    mlir::Value bit16 =
        emitOpaqueCall(rewriter, loc, u16m1Type, vsllU16Callee,
                       mlir::ValueRange{bit, four, vl8}, opName, role);
    return emitOpaqueCall(rewriter, loc, u8mf2Type, vncvtCallee,
                          mlir::ValueRange{bit16, vl8}, opName, role);
  };
  // q5_0 (offsetBias > 0) assembles `((nibble) | (qh_bit << 4)) - bias`; q5_1
  // (offsetBias == 0, the bias ABSENT sentinel) assembles the UNSIGNED 5-bit weight
  // `(nibble) | (qh_bit << 4)` in [0,31] with NO centering vsub (the asymmetric bias
  // lives in the separate per-block MIN fold), byte-identical to the q5_1 direct
  // emitter's assemble5Unsigned (or + reinterpret only).
  auto assemble5 = [&](mlir::Value nibbleU8, mlir::Value bit16) -> mlir::Value {
    mlir::Value a =
        emitOpaqueCall(rewriter, loc, u8mf2Type, orCallee,
                       mlir::ValueRange{nibbleU8, bit16, vl8}, opName, role);
    mlir::Value as =
        emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                       mlir::ValueRange{a}, opName, role);
    if (offsetBias == 0)
      return as;
    return emitOpaqueCall(rewriter, loc, i8mf2Type, subCallee,
                          mlir::ValueRange{as, biasLit, vl8}, opName, role);
  };

  // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k]:
  // *(const int8_t *)(ab + 8 + k).
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
                          mlir::ValueRange{acc, scalar, vec, vl8}, opName, role);
  };

  // ===== q8_0 FULL-int8 core (fullI8), GEMM ONE-strip N-column form: the SIMPLEST
  // flat integer dot -- NO nibble unpack, NO lo/hi split. Per position i in [0,qk):
  // reuse the SHARED signed-i8 strip load (loadNibbles = vle8 i8 at qs[i*16 + roff],
  // NO decode) + per interleaved activation column c in [cLo,cHi) the SHARED plain
  // q8_0x4 scalar read (i8Read al.qs[i*4 + c]), vwmul (i8xi8 -> i16), then vwadd_wv
  // into the column's i32 IN-BLOCK accumulator (full int8 products overflow i16, so
  // NO i16 vwmacc + lo/hi combine). Returns the per-column i32 sumi directly. This
  // is NET-NEW construction (no q8_0 GEMM direct emitter ever existed), validated by
  // the INDEPENDENT oracle (GEMM sumi == GEVM sumi under the x4 activation). =====
  if (fullI8) {
    int64_t positions = nibbleBytes * 2;  // qk (32)
    std::string vwmulCalleeQ8 = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmulQ8 = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCalleeQ8,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCalleeQ8 = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddwQ8 = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCalleeQ8,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };
    std::string mvCalleeQ8 = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    llvm::SmallVector<mlir::Value> sumiVar(activationInterleave);
    for (int64_t c = cLo; c < cHi; ++c) {
      auto v = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32m2Type),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvCalleeQ8, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
      rewriter.create<emitc::AssignOp>(loc, v, seed);
      sumiVar[c] = v;
    }
    auto posLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), sizeLit(positions), sizeLit(1),
        /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard ng(rewriter);
      rewriter.setInsertionPointToStart(posLoop.getBody());
      mlir::Value i = posLoop.getInductionVar();
      step("weight_quant_addr");
      mlir::Value i16 = rewriter.create<emitc::MulOp>(
          loc, sizeType, i, sizeLit(weightInterleave));
      mlir::Value qsOff = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(weightQuantOffset), i16);
      mlir::Value wByteOff =
          rewriter.create<emitc::AddOp>(loc, sizeType, qsOff, roff);
      mlir::Value wStrip = loadNibbles(bl, wByteOff);
      mlir::Value i4 = rewriter.create<emitc::MulOp>(
          loc, sizeType, i, sizeLit(activationInterleave));
      for (int64_t c = cLo; c < cHi; ++c) {
        step("act_quant_addr");
        mlir::Value idx =
            rewriter.create<emitc::AddOp>(loc, sizeType, i4, sizeLit(c));
        mlir::Value aOff = rewriter.create<emitc::AddOp>(
            loc, sizeType, sizeLit(activationQuantOffset), idx);
        mlir::Value aQuant = i8Read(al, aOff);
        mlir::Value prod = vwmulQ8(aQuant, wStrip);
        mlir::Value cur =
            rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar[c], vwaddwQ8(cur, prod));
      }
    }
    llvm::SmallVector<mlir::Value> sumi32(activationInterleave);
    for (int64_t c = cLo; c < cHi; ++c)
      sumi32[c] = rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c])
                      .getResult();
    return sumi32;
  }

  // vint16m1_t sumi_{0..3}_{lo,hi} = vmv_v_x(0, 8);
  std::string mvCallee = riscvIntrinsicName("vmv_v_x", 16, l16, "i16");
  auto seedI16 = [&]() -> mlir::Value {
    return emitOpaqueCallBuilt(
        rewriter, loc, i16m1Type, mvCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zero =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
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

    // b_packed = vle8(&bl.qs[i*16 + roff], 8);  byte = 32 + i*16 + roff
    step("weight_nibble_addr");
    mlir::Value i16 = rewriter.create<emitc::MulOp>(
        loc, sizeType, i, sizeLit(weightInterleave));
    mlir::Value qsOff = rewriter.create<emitc::AddOp>(
        loc, sizeType, sizeLit(weightQuantOffset), i16);
    mlir::Value wByteOff =
        rewriter.create<emitc::AddOp>(loc, sizeType, qsOff, roff);
    mlir::Value packed = loadNibbles(bl, wByteOff);
    mlir::Value bLo, bHi;
    if (hasQh) {
      // q5_0 (hasQh): assemble the 5-bit weight `((nibble) | (qh_bit<<4)) - bias`.
      // The transposed qh masks: low element i at qh+i*2, high element i+16 at
      // qh+(16+i)*2 (== qh + nibbleBytes*2 + i*2); the strip lanes are selected by
      // the RUNTIME roff.
      step("qh_lo_addr");
      mlir::Value iTwo =
          rewriter.create<emitc::MulOp>(loc, sizeType, i, sizeLit(2));
      mlir::Value qhLoOff = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(weightQhByteOffset), iTwo);
      step("qh_hi_addr");
      mlir::Value qhHiBase = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(weightQhByteOffset),
          sizeLit(nibbleBytes * 2));
      mlir::Value qhHiOff =
          rewriter.create<emitc::AddOp>(loc, sizeType, qhHiBase, iTwo);
      mlir::Value loMaskS = qhMaskScalar(bl, qhLoOff);
      mlir::Value hiMaskS = qhMaskScalar(bl, qhHiOff);
      bLo = assemble5(nibbleLoU8(packed), expandQhBit(loMaskS, roff));
      bHi = assemble5(nibbleHiU8(packed), expandQhBit(hiMaskS, roff));
    } else {
      bLo = decodeLo(packed);
      bHi = decodeHi(packed);
    }

    // i*4 (the activation column-quant stride for the low/high halves).
    mlir::Value i4 = rewriter.create<emitc::MulOp>(
        loc, sizeType, i, sizeLit(activationInterleave));

    for (int64_t c = cLo; c < cHi; ++c) {
      // sumi_c_lo = vwmacc_vx(sumi_c_lo, al.qs[i*4+c], b_lo, 8);
      step("act_quant_addr_lo");
      mlir::Value loIdx =
          rewriter.create<emitc::AddOp>(loc, sizeType, i4, sizeLit(c));
      mlir::Value loOff = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(activationQuantOffset), loIdx);
      mlir::Value aLo = i8Read(al, loOff);
      mlir::Value curLo =
          rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[c])
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, sumiLoVar[c],
                                       vwmacc(curLo, aLo, bLo));

      // sumi_c_hi = vwmacc_vx(sumi_c_hi, al.qs[64+i*4+c], b_hi, 8);
      step("act_quant_addr_hi");
      mlir::Value hiIdx =
          rewriter.create<emitc::AddOp>(loc, sizeType, i4, sizeLit(c));
      mlir::Value hiBase = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(activationQuantOffset),
          sizeLit(activationHighRow));
      mlir::Value hiOff =
          rewriter.create<emitc::AddOp>(loc, sizeType, hiBase, hiIdx);
      mlir::Value aHi = i8Read(al, hiOff);
      mlir::Value curHi =
          rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[c])
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, sumiHiVar[c],
                                       vwmacc(curHi, aHi, bHi));
    }
  }

  // const vint32m2_t sumi_c = vwadd_vv(sumi_c_lo, sumi_c_hi, 8);
  std::string vwaddCallee = ("__riscv_vwadd_vv_i32" + l32).str();
  llvm::SmallVector<mlir::Value> sumi32(activationInterleave);
  for (int64_t c = cLo; c < cHi; ++c) {
    mlir::Value lo =
        rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[c])
            .getResult();
    mlir::Value hi =
        rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[c])
            .getResult();
    sumi32[c] = emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                               mlir::ValueRange{lo, hi, vl8}, opName, role);
  }
  return sumi32;
}

// The shared q4_0 16x1-REPACKED GEMM per-block per-column dual-fp16 scale FOLD
// leaf. Factored VERBATIM out of the now-RETIRED emitRepackGemmQ4_0Q8_0 monolith's
// block-loop scale-fold tail so the SAME node sequence (vle16 the per-strip weight
// scales -> per column _Float16 act scale / vfwmul / vfcvt / vfmacc into the column
// f32 accumulator) is driven by the first-class GEMM scale-fold brick inside the typed
// weft_rvv.typed_repack_gemm_loop_body region (emitTypedRepackGemmLoopBody) -- the SOLE
// caller now the monolith is retired, byte-identity by construction preserved. Given
// the per-block bases bl/al,
// the runtime strip row offset roff, the per-column i32 `sumi`, and the per-column
// f32 accumulator lvalues `sumfVar`, it folds each column's sumi into its
// accumulator in place.
void VariantToEmitCFunc::emitRepackGemmDualFp16ScaleFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const RepackGemmDualFp16ScaleFoldContext &cx, mlir::Value bl, mlir::Value al,
    mlir::Value roff, llvm::ArrayRef<mlir::Value> sumi32,
    llvm::ArrayRef<mlir::Value> sumfVar, int64_t cLo, int64_t cHi) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::StringRef opName = cx.opName;
  llvm::StringRef role = cx.role;
  llvm::StringRef l16 = cx.l16;
  llvm::StringRef l32 = cx.l32;
  mlir::Type sizeType = cx.sizeType;
  mlir::Value vl8 = cx.vl8;
  // q4_1 single MIN-fold facts (>= 0 pair => per-column `acc += m_x*s_y[c]`).
  bool hasMin = cx.weightMinByteOffset >= 0 && cx.activationSumByteOffset >= 0;
  int64_t weightMinOffset = cx.weightMinByteOffset;
  int64_t activationSumOffset = cx.activationSumByteOffset;

  mlir::Type f32m2Type =
      emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
  mlir::Type f16m1Type =
      emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
  mlir::Type weightPtrType = bl.getType();
  mlir::Type activationPtrType = al.getType();
  mlir::Type f16PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto step = [&](llvm::StringRef s) {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
  };

  std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
  // vfloat16m1_t b_d = vle16(&bl.d[roff], 8);  byte = roff*2.
  step("weight_scale_addr");
  mlir::Value dByteOff =
      rewriter.create<emitc::MulOp>(loc, sizeType, roff, sizeLit(2));
  mlir::Value dFull =
      rewriter.create<emitc::AddOp>(loc, weightPtrType, bl, dByteOff);
  mlir::Value dCast =
      rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
  mlir::Value bD = emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                                  mlir::ValueRange{dCast, vl8}, opName, role);
  // q4_1: the per-row fp16 MIN strip m_x = vle16(&bl.m[roff]); byte =
  // weightMinOffset + roff*2 (loaded ONCE per strip, folded per column below).
  mlir::Value bM;
  if (hasMin) {
    mlir::Value mByteOff = rewriter.create<emitc::AddOp>(
        loc, sizeType, sizeLit(weightMinOffset), dByteOff);
    mlir::Value mFull =
        rewriter.create<emitc::AddOp>(loc, weightPtrType, bl, mByteOff);
    mlir::Value mCast =
        rewriter.create<emitc::CastOp>(loc, f16PtrType, mFull).getResult();
    bM = emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                        mlir::ValueRange{mCast, vl8}, opName, role);
  }

  // d_c = vfwmul_vf(b_d, *(const _Float16 *)&al.d[c], 8);  -- the raw
  // _Float16 activation scale (NO float cast).
  std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
  std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
  std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
  std::string vfaddCallee = ("__riscv_vfadd_vv_f32" + l32).str();
  llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
  for (int64_t c = cLo; c < cHi; ++c) {
    mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
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
    mlir::Value dC = emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                                    mlir::ValueRange{bD, aD, vl8}, opName, role);
    // sumf_c = vfmacc_vv(sumf_c, vfcvt_f_x_v(sumi_c, 8), d_c, 8);
    mlir::Value sumiF =
        emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                       mlir::ValueRange{sumi32[c], vl8}, opName, role);
    mlir::Value curF =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c]).getResult();
    mlir::Value nextF =
        emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                       mlir::ValueRange{curF, sumiF, dC, vl8}, opName, role);
    // q4_1: the per-column MIN term m_c = vfwmul_vf(b_m, s_y[c]); sumf_c =
    // vfadd_vv(sumf_c, m_c) -- s_y at activationSumOffset + c*2.
    if (hasMin) {
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
      mlir::Value mC =
          emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                         mlir::ValueRange{bM, aS, vl8}, opName, role);
      nextF = emitOpaqueCall(rewriter, loc, f32m2Type, vfaddCallee,
                             mlir::ValueRange{nextF, mC, vl8}, opName, role);
    }
    rewriter.create<emitc::AssignOp>(loc, sumfVar[c], nextF);
  }
}

mlir::LogicalResult VariantToEmitCFunc::emitPackQ4_0ToX16(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlPackQ40ToX16Op pack;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto p = llvm::dyn_cast<weftrvv::GgmlPackQ40ToX16Op>(op))
        pack = p;
    }
    if (!pack)
      return rewriter.notifyMatchFailure(scope, "pack body missing op");

    mlir::Value src = valueMap.lookup(pack.getSrc());
    mlir::Value dst = valueMap.lookup(pack.getDst());
    mlir::Value nblocks = valueMap.lookup(pack.getNblocks());
    if (!src || !dst || !nblocks)
      return rewriter.notifyMatchFailure(pack, "pack ABI operand unmapped");

    llvm::StringRef opName = pack.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = pack.getWEFTEmitCLowerableSourceRole();

    // The plain q4_0 -> q4_0x16 PACK structural facts (I4 mirror, pinned by the
    // verifier): QK=32, block_q4_0 source stride 18 (1 inline fp16 scale @+0, 16
    // nibble bytes @+2), block_q4_0x16 destination stride 288 (16 inline fp16
    // scales @0..32, 256 interleaved nibble bytes @32..288), 16 source blocks
    // interleaved per output block, offset-binary XOR mask 0x88. The transform
    // is the live make_block_q4_0x16 blck_size_interleave==1 branch: pure scalar
    // byte gather + XOR, NO vector machinery.
    int64_t srcStride = pack.getSrcBlockStride();          // 18
    int64_t dstStride = pack.getDstBlockStride();          // 288
    int64_t srcQuantOff = pack.getSrcQuantByteOffset();    // 2
    int64_t dstQuantOff = pack.getDstQuantByteOffset();    // 32
    int64_t interleave = pack.getWeightInterleave();       // 16
    int64_t xorMask = pack.getXorMask();                   // 0x88
    int64_t qk = pack.getQk();                             // 32
    int64_t nibbleBytes = qk / 2;                          // 16 (qs bytes/block)
    int64_t scaleBytes = srcQuantOff;                      // 2 (fp16 d bytes)

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(src) ||
        !llvm::isa<mlir::TypedValue<emitc::PointerType>>(dst))
      return rewriter.notifyMatchFailure(pack, "pack src/dst not pointers");
    auto srcPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(src);
    auto dstPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(dst);
    // The byte rvalue types follow the pointer pointees: src is `const uint8_t*`
    // (pointee `const uint8_t`), dst is `uint8_t*` (pointee `uint8_t`). The XOR
    // literal and the bitwise_xor result carry the dst element type so the store
    // assign and the xor verify cleanly.
    mlir::Type srcEltType = srcPtr.getType().getPointee();
    mlir::Type dstEltType = dstPtr.getType().getPointee();

    // Read one source byte src[idx] (idx a size_t value) as a const-byte rvalue.
    auto srcByte = [&](mlir::Value idx) -> mlir::Value {
      emitc::SubscriptOp sub =
          rewriter.create<emitc::SubscriptOp>(loc, srcPtr, idx);
      return rewriter.create<emitc::LoadOp>(loc, srcEltType, sub.getResult())
          .getResult();
    };
    // Write value (a uint8_t rvalue) to dst[idx].
    auto dstStore = [&](mlir::Value idx, mlir::Value value) {
      emitc::SubscriptOp sub =
          rewriter.create<emitc::SubscriptOp>(loc, dstPtr, idx);
      rewriter.create<emitc::AssignOp>(loc, sub.getResult(), value);
    };
    // idx = a + b
    auto add = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::AddOp>(loc, sizeType, a, b);
    };
    // idx = a * lit
    auto mul = [&](mlir::Value a, int64_t lit) -> mlir::Value {
      return rewriter.create<emitc::MulOp>(loc, sizeType, a, sizeLit(lit));
    };

    // The pack walks each output block b in [0, nblocks): gather the 16
    // consecutive source block_q4_0 (one block from each interleaved column) and
    // emit one block_q4_0x16.
    //
    //   for (size_t b = 0; b < nblocks; b++) {
    //     size_t sbase = b*16*18;  size_t dbase = b*288;
    //     // scales (verbatim copy, NO xor):  out.d[j] = in[j].d  (2 bytes each)
    //     for (size_t j = 0; j < 16; j++)
    //       for (size_t k = 0; k < 2; k++)
    //         dst[dbase + j*2 + k] = src[sbase + j*18 + k];
    //     // quants (16-way interleave + ^0x88):  out.qs[off*16+blk] =
    //     //   in[blk].qs[off] ^ 0x88   (block-major-within-byte)
    //     for (size_t off = 0; off < 16; off++)
    //       for (size_t blk = 0; blk < 16; blk++)
    //         dst[dbase + 32 + off*16 + blk] =
    //           src[sbase + blk*18 + 2 + off] ^ 0x88;
    //   }
    step("pack_block_loop");
    auto blockLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), nblocks, sizeLit(1), /*bodyBuilder=*/nullptr);
    mlir::LogicalResult status = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value b = blockLoop.getInductionVar();
      // size_t sbase = b * (16*18);   size_t dbase = b * 288;
      mlir::Value sbase = mul(b, interleave * srcStride);
      mlir::Value dbase = mul(b, dstStride);

      // Scales: 16 source d (2 bytes each) copied VERBATIM into dst d[16].
      step("pack_scales");
      auto scaleLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(interleave), sizeLit(1),
          /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard sg(rewriter);
        rewriter.setInsertionPointToStart(scaleLoop.getBody());
        mlir::Value j = scaleLoop.getInductionVar();
        // src d at sbase + j*18 ; dst d at dbase + j*2
        mlir::Value sd = add(sbase, mul(j, srcStride));
        mlir::Value dd = add(dbase, mul(j, scaleBytes));
        for (int64_t k = 0; k < scaleBytes; ++k) {
          mlir::Value kLit = sizeLit(k);
          // Cast the const src byte to the dst element type for the store.
          mlir::Value sval = rewriter.create<emitc::CastOp>(
              loc, dstEltType, srcByte(add(sd, kLit)));
          dstStore(add(dd, kLit), sval);
        }
      }

      // Quants: 16-way interleave + ^0x88. out.qs[off*16 + blk] =
      // in[blk].qs[off] ^ 0x88. Outer loop walks the nibble offset (0..16),
      // inner loop walks the 16 interleaved blocks (block-major-within-byte).
      step("pack_quants_xor");
      mlir::Value xorLit = rewriter.create<emitc::LiteralOp>(
          loc, dstEltType, std::to_string(xorMask));
      mlir::Value dstQuantBase = add(dbase, sizeLit(dstQuantOff));
      auto offLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(nibbleBytes), sizeLit(1),
          /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard og(rewriter);
        rewriter.setInsertionPointToStart(offLoop.getBody());
        mlir::Value off = offLoop.getInductionVar();
        // dst quant row base for this offset: dstQuantBase + off*16
        mlir::Value dRowBase = add(dstQuantBase, mul(off, interleave));
        // src byte offset within each block for this nibble: srcQuantOff + off
        mlir::Value sByteOff = add(sizeLit(srcQuantOff), off);
        auto blkLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(interleave), sizeLit(1),
            /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard bg(rewriter);
          rewriter.setInsertionPointToStart(blkLoop.getBody());
          mlir::Value blk = blkLoop.getInductionVar();
          // src index = sbase + blk*18 + (2 + off)
          mlir::Value sIdx = add(add(sbase, mul(blk, srcStride)), sByteOff);
          // dst index = dRowBase + blk
          mlir::Value dIdx = add(dRowBase, blk);
          // Cast the const src byte to the dst element type, then ^0x88.
          mlir::Value packed = rewriter.create<emitc::CastOp>(
              loc, dstEltType, srcByte(sIdx));
          mlir::Value biased = rewriter.create<emitc::BitwiseXorOp>(
              loc, dstEltType, packed, xorLit);
          dstStore(dIdx, biased);
        }
      }
    }
    if (mlir::failed(status))
      return mlir::failure();

    // The pack produces no live value; the result token is the dataflow sink.
    // Materialize a benign 0 literal so the with_vl yield has a mapped value.
    mlir::Value resultTok = sizeLit(0);
    valueMap[pack.getResult()] = resultTok;
    return mlir::success();
  }

// The shared q4_0 16x1-REPACKED per-block LANE-WISE integer CORE leaf. Factored
// VERBATIM out of emitRepackGemvQ4_0Q8_0's block-loop body so the SAME node
// sequence (seed i16 lo/hi per strip -> nibble-step vwmacc loop -> lo/hi vwadd
// combine) is reachable both inline (the monolith) AND through the first-class
// weft_rvv.repack_lane_wise_q4_x_i8_dot brick inside the typed
// weft_rvv.typed_repack_gemv_loop_body region -- byte-identity by construction.
// Given the per-block bases bl/al (already advanced by block_index*stride) it
// returns the per-strip i32 `sumi` values (numHalves entries); the dual-fp16
// per-strip scale fold that consumes them is the caller's.
llvm::SmallVector<mlir::Value>
VariantToEmitCFunc::emitRepackQ4LaneWiseIntegerCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const RepackQ4IntegerCoreContext &cx, mlir::Value bl,
    mlir::Value al) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::StringRef opName = cx.opName;
  llvm::StringRef role = cx.role;
  llvm::StringRef l8 = cx.l8;
  llvm::StringRef l16 = cx.l16;
  llvm::StringRef l32 = cx.l32;
  mlir::Type sizeType = cx.sizeType;
  mlir::Value vl8 = cx.vl8;
  int64_t numHalves = cx.numHalves;
  int64_t half = cx.half;
  int64_t nibbleBytes = cx.nibbleBytes;
  int64_t weightInterleave = cx.weightInterleave;
  int64_t weightQuantOffset = cx.weightQuantOffset;
  int64_t activationQuantOffset = cx.activationQuantOffset;
  int64_t activationHighRow = cx.activationHighRow;

  bool unsignedNibble = cx.unsignedNibble;
  bool hasQh = cx.hasQh;                       // q5_0 5th-bit decode
  bool fullI8 = cx.fullI8;                      // q8_0 full-int8 decode
  int64_t weightQhByteOffset = cx.weightQhByteOffset;
  int64_t offsetBias = cx.offsetBias;
  mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
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
  mlir::Type immI32Type = emitc::OpaqueType::get(ctx, "int");
  mlir::Type i8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
  mlir::Type u8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
  mlir::Type u16PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
  mlir::Type weightPtrType = bl.getType();
  mlir::Type activationPtrType = al.getType();

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto step = [&](llvm::StringRef s) {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
  };

  // q4_0: a typed i8 sub-load __riscv_vle8_v_i8<l8> (the repacked nibbles carry the
  // ^0x88 offset-binary bias). q4_1 (unsignedNibble): the asymmetric weight stores
  // RAW nibbles, so it loads unsigned __riscv_vle8_v_u8. q8_0 (fullI8): the SAME
  // signed vle8 i8 load (unsignedNibble is false), reused as the full-int8 strip.
  std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
  std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
  auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
    mlir::Value full =
        rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
    if (unsignedNibble) {
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    }
    mlir::Value cast =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
    return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                          mlir::ValueRange{cast, vl8}, opName, role);
  };
  // q4_0 offset-binary decode = plain sign-extension: b_lo = vsra(vsll(b,4),4);
  // b_hi = vsra(b,4). q4_1 UNSIGNED decode = the RAW-nibble peel: b_lo =
  // vreinterpret_i8(vand(b,0x0F)); b_hi = vreinterpret_i8(vsrl(b,4)) (value-identity
  // for 0..15 -- the q4_1 bias lives in the separate MIN scale, NO sign-extend).
  std::string sllCallee = ("__riscv_vsll_vx_i8" + l8).str();
  std::string sraCallee = ("__riscv_vsra_vx_i8" + l8).str();
  std::string vandCallee = ("__riscv_vand_vx_u8" + l8).str();
  std::string vsrlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
  std::string reinterpretCallee =
      ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
  mlir::Value four = sizeLit(4);
  auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
    if (unsignedNibble) {
      mlir::Value lo = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vandCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value mask =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x0F")
                    .getResult();
            return {packed, mask, vl8};
          });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{lo}, opName, role);
    }
    mlir::Value shl =
        emitOpaqueCall(rewriter, loc, i8mf2Type, sllCallee,
                       mlir::ValueRange{packed, four, vl8}, opName, role);
    return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                          mlir::ValueRange{shl, four, vl8}, opName, role);
  };
  auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
    if (unsignedNibble) {
      mlir::Value hi = emitOpaqueCallBuilt(
          rewriter, loc, u8mf2Type, vsrlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sh =
                rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x04")
                    .getResult();
            return {packed, sh, vl8};
          });
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                            mlir::ValueRange{hi}, opName, role);
    }
    return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                          mlir::ValueRange{packed, four, vl8}, opName, role);
  };

  // ===== q5_0 5th-bit (qh) decode leaf (hasQh) =====================
  // The five-bit weight is `A = nibble | (qh_bit << 4)` in [0,31], reinterpreted
  // u8->i8, then `-offsetBias` (16) -- the SAME reconstruct as the retired q5_0
  // direct emitter (fifthBitLane + reinterpretBias), off the RAW unsigned nibble
  // peel. The qh masks are read once per element step and each strip selects its
  // lanes via the (vid + h*half) shift.
  std::string orCallee = ("__riscv_vor_vv_u8" + l8).str();
  std::string subCallee = ("__riscv_vsub_vx_i8" + l8).str();
  mlir::Value biasLit = sizeLit(offsetBias);
  // lo nibble [0,15] unsigned: vand(b, 0x0F).
  auto nibbleLoU8 = [&](mlir::Value packed) -> mlir::Value {
    return emitOpaqueCallBuilt(
        rewriter, loc, u8mf2Type, vandCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {packed, sizeLit(15), vl8};
        });
  };
  // hi nibble [0,15] unsigned: vsrl(b, 4).
  auto nibbleHiU8 = [&](mlir::Value packed) -> mlir::Value {
    return emitOpaqueCall(rewriter, loc, u8mf2Type, vsrlCallee,
                          mlir::ValueRange{packed, four, vl8}, opName, role);
  };
  // ===== REDESIGN-B (G7 L2 qh-plane) native-mask 5-bit decode ==================
  // The transposed qh u16-per-step layout IS a per-lane bool plane already: for
  // element step i, byte (h*half/8) of the step-i u16 holds strip h's `half` mask
  // bits, lane l = row (l + h*half)'s 5th bit -- BIT-IDENTICAL to the retired
  // per-lane (vid + h*half) select. So load the strip mask DIRECTLY with vlm and
  // fuse the 5th-bit selection + the offset-binary bias into ONE masked op (stock
  // ggml_vec_dot_q5_0_q8_0's native-mask trick), retiring the OLD
  // splat/vid/vsrl_vv/vand/vsll/vncvt/vor per-lane expand chain:
  //   q5_0 (offsetBias > 0):  weight   = qh_bit ? nibble : nibble - bias
  //                         = vsub_vx_i8_mu(!qh_mask, nib_i8, nib_i8, bias) [invert]
  //   q5_1 (offsetBias == 0): weight_u = qh_bit ? nibble + 16 : nibble
  //                         = vadd_vx_u8_mu(qh_mask, nib_u8, nib_u8, 16)    [raw mask]
  // Algebraic identity `(nibble | (bit<<4)) - 16 == bit ? nibble : nibble - 16`
  // (and its unsigned q5_1 sibling) -- byte-exact by construction (see the
  // qh-plane G1 casefile raw/byteexact_model.c: 0/32 both arms). The nibble
  // extract + the vwmacc accumulate order are UNCHANGED. The predicate width
  // vbool<N> for an i8<l8> vector is N = 8/LMUL (mf2 -> b16).
  unsigned qhMaskBits = 16;
  if (l8 == "mf8") qhMaskBits = 64;
  else if (l8 == "mf4") qhMaskBits = 32;
  else if (l8 == "mf2") qhMaskBits = 16;
  else if (l8 == "m1") qhMaskBits = 8;
  else if (l8 == "m2") qhMaskBits = 4;
  else if (l8 == "m4") qhMaskBits = 2;
  else if (l8 == "m8") qhMaskBits = 1;
  std::string maskBitsStr = std::to_string(qhMaskBits);
  mlir::Type qhBoolType =
      emitc::OpaqueType::get(ctx, ("vbool" + maskBitsStr + "_t"));
  std::string vlmCallee = "__riscv_vlm_v_b" + maskBitsStr;
  std::string vmnandCallee = "__riscv_vmnand_mm_b" + maskBitsStr;
  std::string subMuCallee = ("__riscv_vsub_vx_i8" + l8 + "_mu").str();
  std::string addMuCallee = ("__riscv_vadd_vx_u8" + l8 + "_mu").str();
  // A mask byte covers 8 lanes; strip h's mask sits h*(half/8) bytes further in
  // the step-i u16 (byte-aligned since half is a multiple of 8).
  int64_t qhStripMaskBytes = half / 8;
  // Load strip h's qh mask bits directly (byte-aligned vlm, lane l = bit l).
  auto loadQhMaskBits = [&](mlir::Value qhOff, int64_t h) -> mlir::Value {
    mlir::Value off = qhOff;
    if (h != 0)
      off = rewriter.create<emitc::AddOp>(loc, sizeType, qhOff,
                                          sizeLit(h * qhStripMaskBytes));
    mlir::Value full =
        rewriter.create<emitc::AddOp>(loc, weightPtrType, bl, off);
    mlir::Value cast =
        rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
    return emitOpaqueCall(rewriter, loc, qhBoolType, vlmCallee,
                          mlir::ValueRange{cast, vl8}, opName, role,
                          llvm::StringRef("qh_mask_bits"));
  };
  // Fuse the 5th-bit + bias into ONE masked op off the RAW unsigned nibble.
  auto decodeQh5 = [&](mlir::Value nibbleU8, mlir::Value qhOff,
                       int64_t h) -> mlir::Value {
    mlir::Value mask = loadQhMaskBits(qhOff, h);
    if (offsetBias != 0) {
      // q5_0: invert the 5th-bit mask, masked-vsub the -bias on the UNSET lanes.
      mlir::Value inv =
          emitOpaqueCall(rewriter, loc, qhBoolType, vmnandCallee,
                         mlir::ValueRange{mask, mask, vl8}, opName, role);
      mlir::Value nibI8 =
          emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                         mlir::ValueRange{nibbleU8}, opName, role);
      return emitOpaqueCall(
          rewriter, loc, i8mf2Type, subMuCallee,
          mlir::ValueRange{inv, nibI8, nibI8, biasLit, vl8}, opName, role);
    }
    // q5_1: masked-vadd +16 on the SET lanes (raw mask, NO invert), reinterpret.
    mlir::Value added = emitOpaqueCallBuilt(
        rewriter, loc, u8mf2Type, addMuCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {mask, nibbleU8, nibbleU8, sizeLit(16), vl8};
        });
    return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpretCallee,
                          mlir::ValueRange{added}, opName, role);
  };

  // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k]:
  // *(const int8_t *)(ab + 2 + k).
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

  // ===== q8_0 FULL-int8 core (cx.fullI8): the SIMPLEST flat integer dot -- NO
  // nibble unpack, NO lo/hi split. The repacked q8_0 weight bytes are FULL signed
  // int8 (one weight per contraction position; qk == nibbleBytes*2 == 32 positions
  // per block, each reading ALL 16 lanes of a strip at 32 + i*16 + h*half). Per
  // position: reuse the SHARED signed-i8 strip load (loadNibbles at unsignedNibble
  // == false is exactly a vle8 i8 load, NO decode) + the SHARED plain q8_0
  // activation scalar read (i8Read al.qs[i]), vwmul (i8xi8 -> i16), then vwadd_wv
  // into an i32 IN-BLOCK accumulator (full int8 products overflow i16 after 3 terms,
  // so the q4_0 i16 vwmacc + end-of-block vwadd combine is REPLACED by i32 in-block
  // accumulation). Byte-exact to the retired emitRepackGemvQ8_0Q8_0 integer part.
  // Returns the numHalves per-strip i32 sumi directly (no lo/hi combine). =====
  if (fullI8) {
    int64_t positions = nibbleBytes * 2;  // qk (32)
    std::string vwmulCalleeQ8 = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmulQ8 = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCalleeQ8,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    std::string vwaddwCalleeQ8 = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddwQ8 = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCalleeQ8,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };
    std::string mvCalleeQ8 = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
    llvm::SmallVector<mlir::Value> sumiVar;
    for (int64_t h = 0; h < numHalves; ++h) {
      auto v = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32m2Type),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, mvCalleeQ8, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, vl8};
          });
      rewriter.create<emitc::AssignOp>(loc, v, seed);
      sumiVar.push_back(v);
    }
    auto posLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), sizeLit(positions), sizeLit(1),
        /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard ng(rewriter);
      rewriter.setInsertionPointToStart(posLoop.getBody());
      mlir::Value i = posLoop.getInductionVar();
      step("weight_quant_addr");
      mlir::Value i16 = rewriter.create<emitc::MulOp>(
          loc, sizeType, i, sizeLit(weightInterleave));
      mlir::Value qsOff = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(weightQuantOffset), i16);
      llvm::SmallVector<mlir::Value> wByteOff;
      for (int64_t h = 0; h < numHalves; ++h) {
        if (h == 0)
          wByteOff.push_back(qsOff);
        else
          wByteOff.push_back(rewriter.create<emitc::AddOp>(
              loc, sizeType, qsOff, sizeLit(h * half)));
      }
      llvm::SmallVector<mlir::Value> wStrip;
      for (int64_t h = 0; h < numHalves; ++h)
        wStrip.push_back(loadNibbles(bl, wByteOff[h]));
      step("act_quant_addr");
      mlir::Value aOff = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(activationQuantOffset), i);
      mlir::Value aQuant = i8Read(al, aOff);
      for (int64_t h = 0; h < numHalves; ++h) {
        mlir::Value prod = vwmulQ8(aQuant, wStrip[h]);
        mlir::Value cur =
            rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar[h], vwaddwQ8(cur, prod));
      }
    }
    llvm::SmallVector<mlir::Value> sumi;
    for (int64_t h = 0; h < numHalves; ++h)
      sumi.push_back(rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                         .getResult());
    return sumi;
  }

  // vint16m1_t sumi_{a,b}_{lo,hi} = vmv_v_x(0, 8);
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
  // Per-strip i16 lo/hi accumulator lvalues, seeded lo then hi per strip:
  // at num_halves=2 this is a_lo, a_hi, b_lo, b_hi (HEAD order, byte-id).
  llvm::SmallVector<mlir::Value> sumiLoVar, sumiHiVar;
  for (int64_t h = 0; h < numHalves; ++h) {
    auto vlo = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(i16m1Type),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(loc, vlo, seedI16());
    sumiLoVar.push_back(vlo);
    auto vhi = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(i16m1Type),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(loc, vhi, seedI16());
    sumiHiVar.push_back(vhi);
  }

  // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
  auto nibLoop = rewriter.create<emitc::ForOp>(
      loc, sizeLit(0), sizeLit(nibbleBytes), sizeLit(1),
      /*bodyBuilder=*/nullptr);
  {
    mlir::OpBuilder::InsertionGuard ng(rewriter);
    rewriter.setInsertionPointToStart(nibLoop.getBody());
    mlir::Value i = nibLoop.getInductionVar();

    // num_halves disjoint contiguous sub-loads, strip h at qs[i*16+h*half]:
    // rows 0..7 at qs[i*16+0] and rows 8..15 at qs[i*16+8] (half=8, 2
    // strips), or rows 0..15 at qs[i*16+0] (half=16, 1 strip). byte =
    // 32 + i*16 (+ h*half). LOAD phase first (FileCheck pins both vle8
    // before any decode); the h=0 offset is qsOff with NO AddOp (matches
    // HEAD's wByteOffA), h>0 adds h*half (matches HEAD's wByteOffB).
    step("weight_nibble_addr");
    mlir::Value i16 = rewriter.create<emitc::MulOp>(
        loc, sizeType, i, sizeLit(weightInterleave));
    mlir::Value qsOff = rewriter.create<emitc::AddOp>(
        loc, sizeType, sizeLit(weightQuantOffset), i16);
    // Compute every strip's byte offset FIRST (HEAD computed wByteOffA =
    // qsOff and wByteOffB = qsOff+8 before either vle8), then issue all
    // loads -- preserves HEAD's exact node order at num_halves=2. The h=0
    // offset is qsOff with NO AddOp (matches HEAD's wByteOffA).
    llvm::SmallVector<mlir::Value> wByteOff;
    for (int64_t h = 0; h < numHalves; ++h) {
      if (h == 0)
        wByteOff.push_back(qsOff);
      else
        wByteOff.push_back(rewriter.create<emitc::AddOp>(
            loc, sizeType, qsOff, sizeLit(h * half)));
    }
    llvm::SmallVector<mlir::Value> packed;
    for (int64_t h = 0; h < numHalves; ++h)
      packed.push_back(loadNibbles(bl, wByteOff[h]));
    // DECODE phase: per strip, lo then hi. q4_0/q4_1 (4-bit): plain
    // sign-extension / unsigned peel. q5_0/q5_1 (hasQh): REDESIGN-B native-mask
    // 5-bit assembly -- the transposed qh mask bits are loaded DIRECTLY per strip
    // (low element i at qh+i*2, high element i+16 at qh+(16+i)*2; strip h at
    // byte-offset h*(half/8)) and the 5th-bit + bias are fused into ONE masked op.
    llvm::SmallVector<mlir::Value> bLo, bHi;
    if (hasQh) {
      step("qh_lo_addr");
      mlir::Value iTwo =
          rewriter.create<emitc::MulOp>(loc, sizeType, i, sizeLit(2));
      mlir::Value qhLoOff = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(weightQhByteOffset), iTwo);
      step("qh_hi_addr");
      mlir::Value qhHiBase = rewriter.create<emitc::AddOp>(
          loc, sizeType, sizeLit(weightQhByteOffset),
          sizeLit(activationHighRow * 2));
      mlir::Value qhHiOff =
          rewriter.create<emitc::AddOp>(loc, sizeType, qhHiBase, iTwo);
      for (int64_t h = 0; h < numHalves; ++h) {
        bLo.push_back(decodeQh5(nibbleLoU8(packed[h]), qhLoOff, h));
        bHi.push_back(decodeQh5(nibbleHiU8(packed[h]), qhHiOff, h));
      }
    } else {
      for (int64_t h = 0; h < numHalves; ++h) {
        bLo.push_back(decodeLo(packed[h]));
        bHi.push_back(decodeHi(packed[h]));
      }
    }

    // Single activation column (SHARED across strips, read ONCE -- this is
    // a GEMV, one column): low quant al.qs[i], high quant al.qs[16+i].
    // byte = 2 + i (low), 2 + 16 + i (high).
    step("act_quant_addr_lo");
    mlir::Value loOff = rewriter.create<emitc::AddOp>(
        loc, sizeType, sizeLit(activationQuantOffset), i);
    mlir::Value aLo = i8Read(al, loOff);
    step("act_quant_addr_hi");
    mlir::Value hiBase = rewriter.create<emitc::AddOp>(
        loc, sizeType, sizeLit(activationQuantOffset),
        sizeLit(activationHighRow));
    mlir::Value hiOff =
        rewriter.create<emitc::AddOp>(loc, sizeType, hiBase, i);
    mlir::Value aHi = i8Read(al, hiOff);

    // ACCUMULATE phase: per strip, lane-wise vwmacc lo then hi. At half=8
    // this is a_lo, a_hi, b_lo, b_hi (HEAD order, byte-identical).
    for (int64_t h = 0; h < numHalves; ++h) {
      // sumi_h_lo = vwmacc(sumi_h_lo, al.qs[i],    b_h_lo, vl);
      mlir::Value curLo =
          rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[h])
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, sumiLoVar[h],
                                       vwmacc(curLo, aLo, bLo[h]));
      // sumi_h_hi = vwmacc(sumi_h_hi, al.qs[16+i], b_h_hi, vl);
      mlir::Value curHi =
          rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[h])
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, sumiHiVar[h],
                                       vwmacc(curHi, aHi, bHi[h]));
    }
  }

  // const vint32m2_t sumi_a = vwadd_vv(sumi_a_lo, sumi_a_hi, 8);  and _b.
  std::string vwaddCallee = ("__riscv_vwadd_vv_i32" + l32).str();
  auto combine = [&](mlir::Value loVar, mlir::Value hiVar) -> mlir::Value {
    mlir::Value lo =
        rewriter.create<emitc::LoadOp>(loc, i16m1Type, loVar).getResult();
    mlir::Value hi =
        rewriter.create<emitc::LoadOp>(loc, i16m1Type, hiVar).getResult();
    return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                          mlir::ValueRange{lo, hi, vl8}, opName, role);
  };
  // Per-strip lo/hi combine: at half=8 this is sumi_a then sumi_b (HEAD).
  llvm::SmallVector<mlir::Value> sumi;
  for (int64_t h = 0; h < numHalves; ++h)
    sumi.push_back(combine(sumiLoVar[h], sumiHiVar[h]));
  return sumi;
}

// The shared q4_0 16x1-REPACKED per-block per-strip dual-fp16 scale FOLD leaf.
// Factored VERBATIM out of emitRepackGemvQ4_0Q8_0's block-loop scale-fold tail so
// the SAME node sequence (per-strip vle16 weight scale -> ONE _Float16 activation
// scale -> per-strip vfwmul/vfcvt/vfmacc into the f32 accumulator) is reachable
// both inline (the monolith) AND through the first-class
// weft_rvv.repack_dual_fp16_scale_fold brick inside the typed
// weft_rvv.typed_repack_gemv_loop_body region -- byte-identity by construction.
// Given the per-block bases bl/al (already advanced by block_index*stride), the
// per-strip i32 `sumi` from the integer core, and the per-strip f32 accumulator
// lvalues `sumfVar`, it folds sumi into each accumulator in place.
void VariantToEmitCFunc::emitRepackDualFp16ScaleFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const RepackDualFp16ScaleFoldContext &cx, mlir::Value bl, mlir::Value al,
    llvm::ArrayRef<mlir::Value> sumi,
    llvm::ArrayRef<mlir::Value> sumfVar) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::StringRef opName = cx.opName;
  llvm::StringRef role = cx.role;
  llvm::StringRef l16 = cx.l16;
  llvm::StringRef l32 = cx.l32;
  mlir::Type sizeType = cx.sizeType;
  mlir::Value vl8 = cx.vl8;
  int64_t numHalves = cx.numHalves;
  int64_t half = cx.half;
  int64_t weightScaleOffset = cx.weightScaleByteOffset;
  int64_t activationScaleOffset = cx.activationScaleByteOffset;
  // q4_1 single MIN-fold facts (>= 0 pair => the lane-wise `acc += m_x*s_y`
  // correction after the dual-fp16 scale fold; -1 => q4_0 no-min, byte-identical).
  bool hasMin = cx.weightMinByteOffset >= 0 && cx.activationSumByteOffset >= 0;
  int64_t weightMinOffset = cx.weightMinByteOffset;
  int64_t activationSumOffset = cx.activationSumByteOffset;

  mlir::Type f32m2Type =
      emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
  mlir::Type f16m1Type =
      emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
  mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
  mlir::Type f16PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));
  mlir::Type weightPtrType = bl.getType();

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto step = [&](llvm::StringRef s) {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
  };

  // vfloat16<l16>_t b_d_h = vle16(&bl.d[h*half], vl); one scale strip each. The
  // per-strip byte offset folds the block-leading scale byte offset (0) with the
  // strip lane offset h*half*2 into ONE literal -- byte-exact to HEAD's
  // AddOp-free h==0 first strip.
  std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
  auto loadScales = [&](int64_t laneOff) -> mlir::Value {
    step("weight_scale_addr");
    int64_t totalOff = weightScaleOffset + laneOff * 2;
    mlir::Value dFull = bl;
    if (totalOff != 0)
      dFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                            sizeLit(totalOff));
    mlir::Value dCast =
        rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull).getResult();
    return emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                          mlir::ValueRange{dCast, vl8}, opName, role);
  };
  llvm::SmallVector<mlir::Value> bD;
  for (int64_t h = 0; h < numHalves; ++h)
    bD.push_back(loadScales(h * half));
  // q4_1: the per-row fp16 MIN strips m_x (at weightMinOffset + h*half*2), one per
  // strip, loaded the SAME way as the d strips.
  llvm::SmallVector<mlir::Value> bM;
  if (hasMin)
    for (int64_t h = 0; h < numHalves; ++h) {
      int64_t totalOff = weightMinOffset + h * half * 2;
      mlir::Value mFull = bl;
      if (totalOff != 0)
        mFull = rewriter.create<emitc::AddOp>(loc, weightPtrType, bl,
                                              sizeLit(totalOff));
      mlir::Value mCast =
          rewriter.create<emitc::CastOp>(loc, f16PtrType, mFull).getResult();
      bM.push_back(emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                                  mlir::ValueRange{mCast, vl8}, opName, role));
    }

  // The single activation scale *(const _Float16 *)&al.d (NO float cast),
  // broadcast into both halves' vfwmul. The block-leading scale byte offset (0)
  // is AddOp-free, byte-exact to HEAD's direct al cast.
  llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
  mlir::Value aD = emitOpaqueCallBuilt(
      rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
      [&](mlir::OpBuilder &b,
          mlir::Location l) -> llvm::SmallVector<mlir::Value> {
        mlir::Value aBase = al;
        if (activationScaleOffset != 0)
          aBase = rewriter.create<emitc::AddOp>(loc, al.getType(), al,
                                                sizeLit(activationScaleOffset));
        mlir::Value aDCast =
            rewriter.create<emitc::CastOp>(loc, f16PtrType, aBase).getResult();
        return {aDCast};
      },
      llvm::StringRef("act_scale_scalar"));
  // q4_1: the single activation scaled-sum s_y = *(const _Float16 *)&al.s (at
  // activationSumOffset), broadcast into the lane-wise MIN term m_x * s_y.
  mlir::Value aS;
  if (hasMin)
    aS = emitOpaqueCallBuilt(
        rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value aSFull = rewriter.create<emitc::AddOp>(
              loc, al.getType(), al, sizeLit(activationSumOffset));
          mlir::Value aSCast =
              rewriter.create<emitc::CastOp>(loc, f16PtrType, aSFull)
                  .getResult();
          return {aSCast};
        },
        llvm::StringRef("act_sum_scalar"));

  // d_h = vfwmul_vf(b_d_h, aD, vl);
  // sumf_h = vfmacc_vv(sumf_h, vfcvt_f_x_v(sumi_h, vl), d_h, vl);
  // q4_1 min: m_h = vfwmul_vf(b_m_h, aS, vl); sumf_h = vfadd_vv(sumf_h, m_h, vl).
  std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
  std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
  std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
  std::string vfaddCallee = ("__riscv_vfadd_vv_f32" + l32).str();
  auto fold = [&](mlir::Value bDStrip, mlir::Value bMStrip,
                  mlir::Value sumiStrip, mlir::Value sumfVarStrip) {
    mlir::Value dC =
        emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                       mlir::ValueRange{bDStrip, aD, vl8}, opName, role);
    mlir::Value sumiF =
        emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                       mlir::ValueRange{sumiStrip, vl8}, opName, role);
    mlir::Value curF =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVarStrip)
            .getResult();
    mlir::Value nextF =
        emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                       mlir::ValueRange{curF, sumiF, dC, vl8}, opName, role);
    if (hasMin) {
      mlir::Value mC =
          emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                         mlir::ValueRange{bMStrip, aS, vl8}, opName, role);
      nextF = emitOpaqueCall(rewriter, loc, f32m2Type, vfaddCallee,
                             mlir::ValueRange{nextF, mC, vl8}, opName, role);
    }
    rewriter.create<emitc::AssignOp>(loc, sumfVarStrip, nextF);
  };
  for (int64_t h = 0; h < numHalves; ++h)
    fold(bD[h], hasMin ? bM[h] : mlir::Value(), sumi[h], sumfVar[h]);
}

// M-FLAT REPACK loop-scaffold Phase B (full-body byte-exact, ALL arms): the typed
// region-carrying sibling of the monolithic emitRepackGemvQ4_0Q8_0. It lowers
// weft_rvv.typed_repack_gemv_loop_body -- whose region carries the inner
// contraction-block loop with the numHalves per-strip LANE-WISE f32 VECTOR
// loop-carried accumulators -- to the byte-exact repacked GEVM kernel across EVERY
// resource arm: the VLEN=256 fractional one-strip mf2 form (numHalves==1, f32m2),
// the VLEN=128 two-8-lane-halves mf2 form (numHalves==2, two f32m2), and the RVV0.7
// whole-LMUL one-strip m1 form (numHalves==1, f32m4). nb = n / QK, nc_groups = nc /
// weight_interleave, the outer weight-column-group emitc.for, the numHalves
// per-strip vfloat32{m2,m4} emitc.variable accumulators seeded per group with
// vfmv_v_f(0.0f), the inner block emitc.for, and the per-strip lane-wise vse32
// stores (NO horizontal reduction -- each repacked block-as-lane strip writes its
// 16/8 columns straight to s + x*16 + strip*half). The inner body is FULL-BODY
// byte-exact: the integer CORE brick (weft_rvv.repack_lane_wise_q4_x_i8_dot -> the
// SHARED emitRepackQ4LaneWiseIntegerCore leaf, producing numHalves per-strip sumi)
// is FOLLOWED by the numHalves dual-fp16 scale FOLD bricks
// (weft_rvv.repack_dual_fp16_scale_fold -> the SHARED emitRepackDualFp16ScaleFold
// leaf, called ONCE over all strips: load each weight scale / ONE _Float16 act
// scale / per-strip vfwmul / vfcvt / vfmacc, load acc, fold, assign back). Both
// leaves are shared VERBATIM with the monolith and driven by the SAME numHalves /
// half / l8/l16/l32 / byte-offset facts, so the whole kernel body is byte-identical
// to emitRepackGemvQ4_0Q8_0's by construction on every arm (the monolith's trailing
// unused-result token is its only residue -- the loop-body op has no result).

mlir::LogicalResult VariantToEmitCFunc::emitTypedRepackGemvLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::MLIRContext *ctx = rewriter.getContext();

  weftrvv::TypedRepackGemvLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<weftrvv::TypedRepackGemvLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed repack GEVM loop body missing the op");

  auto requireMainTermRolled = [&]() -> mlir::FailureOr<bool> {
    std::optional<llvm::StringRef> form = loopBody.getMainTermForm();
    if (!form || (*form != "unrolled" && *form != "rolled"))
      return mlir::failure();
    return *form == "rolled";
  };

  // ---- TERNARY front-door dispatch (the retired emitRepackGem{v}TQ{20,10}Q8K
  // direct emitters, now CONSTRUCTED through this typed-region front door). When the
  // loop body carries the ternary fold_model its decomposed in-region brick is the
  // weft_rvv.repack_gemv_ternary_core (NOT the q4_0 nibble core + dual-fp16 fold).
  // We GATE the emit on that brick's block_index-tied + base-tied anti-bypass, then
  // RE-EMIT the byte-exact ternary GEVM body from the shared body leaf (the SAME
  // construction discipline as the flat ternary vec_dot core brick; byte-exactness
  // to the retired direct emitter is by construction). ----
  if (loopBody.getFoldModel() == "ternary_single_fp16_scale") {
    weftrvv::RepackGemvTernaryCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemvTernaryCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "ternary repack GEVM loop body requires the region "
                    "weft_rvv.repack_gemv_ternary_core integer-core brick");
    if (coreBrick.getBlockIndex() !=
        loopBody.getBody().front().getArgument(0))
      return rewriter.notifyMatchFailure(
          coreBrick, "the ternary core brick's block_index must be the loop "
                     "induction variable (region arg 0)");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the ternary core brick's weight/activation bases must be "
                     "the loop-body's own repacked-weight / q8_K-activation ABI "
                     "buffers");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(
          loopBody, "ternary repack GEVM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    if (coreBrick.getDecodeModel() == "tq2_0")
      return emitRepackTernaryGemvBodyTQ20(
          rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    // tq1_0 base-3: the base-3 decode reads a SECOND weight plane (qh), whose
    // repacked byte offset rides on the loop body op's OPTIONAL weight_qh_byte_offset
    // attr (the tq2_0 single-plane fold does not carry it).
    if (coreBrick.getDecodeModel() == "tq1_0") {
      std::optional<uint64_t> qhOff = loopBody.getWeightQhByteOffset();
      if (!qhOff)
        return rewriter.notifyMatchFailure(
            loopBody, "tq1_0 ternary repack GEVM loop body requires the base-3 "
                      "weight_qh_byte_offset SECOND-plane attr");
      return emitRepackTernaryGemvBodyTQ10(
          rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(*qhOff),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    }
    return rewriter.notifyMatchFailure(
        coreBrick, "ternary repack GEVM decode_model not recognized (expected "
                   "\"tq2_0\" 2-bit or \"tq1_0\" base-3)");
  }

  // ---- CODEBOOK front-door dispatch (the retired emitRepackGem{v,m}Iq4{Nl,Xs} direct
  // emitters, now CONSTRUCTED through this typed-region front door). When the loop body
  // carries a codebook fold_model its decomposed in-region brick is the
  // weft_rvv.repack_gemv_codebook_core (decode_model "iq4_nl" flat OR "iq4_xs"
  // super-block). We GATE the emit on that brick's block_index-tied + base-tied
  // anti-bypass, then RE-EMIT the byte-exact codebook GEVM body (the 16-entry MEMORY
  // vluxei16 codebook gather + the i32 dot; iq4_nl: single fp16 scale fold; iq4_xs: the
  // K-quant 6-bit SIGNED per-sub-block scale + i32 vmacc fold) from the shared body leaf;
  // byte-exactness to the retired direct emitter is by construction. The 16-entry
  // non-linear codebook rides the core brick's DenseI8ArrayAttr (the load-bearing WHAT the
  // gather indexes -- the gate proves it is a REAL memory gather, not a fake-linear
  // value). ----
  bool isCodebookFlatFoldV =
      loopBody.getFoldModel() == "codebook_flat_single_scale";
  bool isCodebookSuperblockFoldV =
      loopBody.getFoldModel() == "codebook_superblock_signed6_no_min";
  // The mxfp4 FLAT codebook fold WITH the E8M0 shared-exponent per-column scale (the
  // E8M0 sibling of the iq4_nl flat single-fp16-scale fold; decode_model "mxfp4").
  bool isCodebookE8m0FoldV =
      loopBody.getFoldModel() == "codebook_flat_e8m0_scale";
  if (isCodebookFlatFoldV || isCodebookSuperblockFoldV || isCodebookE8m0FoldV) {
    weftrvv::RepackGemvCodebookCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemvCodebookCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "codebook repack GEVM loop body requires the region "
                    "weft_rvv.repack_gemv_codebook_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0))
      return rewriter.notifyMatchFailure(
          coreBrick, "the codebook core brick's block_index must be the loop "
                     "induction variable (region arg 0)");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the codebook core brick's weight/activation bases must be "
                     "the loop-body's own repacked-weight / q8-activation ABI "
                     "buffers");
    // The decode_model MUST agree with the fold_model: iq4_nl on the flat fold,
    // iq4_xs on the super-block signed-6 fold, mxfp4 on the flat E8M0 fold
    // (fail-closed, I7).
    bool isIq4Xs = coreBrick.getDecodeModel() == "iq4_xs";
    bool isIq4Nl = coreBrick.getDecodeModel() == "iq4_nl";
    bool isMxfp4 = coreBrick.getDecodeModel() == "mxfp4";
    if ((isCodebookSuperblockFoldV && !isIq4Xs) ||
        (isCodebookFlatFoldV && !isIq4Nl) ||
        (isCodebookE8m0FoldV && !isMxfp4))
      return rewriter.notifyMatchFailure(
          coreBrick, "codebook repack GEVM decode_model not recognized / does not "
                     "match the fold_model (expected \"iq4_nl\" on the flat fold, "
                     "\"iq4_xs\" on the super-block signed-6 fold, or \"mxfp4\" on "
                     "the flat E8M0 fold)");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(
          loopBody, "codebook repack GEVM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // The iq4_xs SUPER-BLOCK sibling: the SAME codebook gather + i32 dot PLUS the
    // K-quant 6-bit SIGNED per-sub-block scale fold. Its super-block decode facts
    // (scales_l LOW pair region, scales_h HIGH 2-bit region, sub-block count) ride on
    // the loop body op's OPTIONAL attrs (fail-closed, I7).
    if (isIq4Xs) {
      std::optional<uint64_t> scalesLow = loopBody.getWeightScalesByteOffset();
      std::optional<uint64_t> scalesHigh =
          loopBody.getWeightScalesHighByteOffset();
      std::optional<uint64_t> nSub = loopBody.getNSubblocks();
      if (!scalesLow || !scalesHigh || !nSub)
        return rewriter.notifyMatchFailure(
            loopBody, "iq4_xs codebook repack GEVM loop body requires the "
                      "super-block decode attrs weight_scales_byte_offset / "
                      "weight_scales_high_byte_offset / n_subblocks");
      return emitRepackCodebookGemvBodyIq4Xs(
          rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(*scalesLow), static_cast<int64_t>(*scalesHigh),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*nSub), coreBrick.getCodebook(),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    }
    // The mxfp4 FLAT E8M0 sibling: the SAME codebook gather + i32 dot, but the per-column
    // weight scale is the E8M0 shared-exponent bit-construction (NO fp16 scale, NO min, NO
    // sub-block). Same body-leaf signature as iq4_nl (flat, nSubblocks == 0).
    if (isMxfp4)
      return emitRepackCodebookGemvBodyMxfp4(
          rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          coreBrick.getCodebook(),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    return emitRepackCodebookGemvBodyIq4Nl(
        rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
        sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
        static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
        coreBrick.getCodebook(),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()));
  }

  // ---- GRID front-door dispatch (the retired emitRepackGemvIq2XxsQ8K direct emitter,
  // now CONSTRUCTED through this typed-region front door). When the loop body carries the
  // grid fold_model its decomposed in-region brick is the weft_rvv.repack_gemv_grid_core
  // (decode_model "iq2_xxs", NOT the codebook / K-quant / q4_0 cores). We GATE the emit on
  // that brick's block_index-tied + base-tied anti-bypass, then RE-EMIT the byte-exact
  // iq2_xxs GEVM body (the REAL grid-index vluxei16 GATHER + the sign-plane vluxei16 GATHER
  // + the vmul-onto-grid sign fold + the i32 in-block dot + the per-sub-block int8 ls-scale
  // vmacc fold + the 0.125 store) from the shared body leaf; byte-exactness to the retired
  // direct emitter is by construction. The grid decode facts (grid-index / ls-scale /
  // sign-selector byte offsets + n_subblocks) ride on the grid core brick; the FIXED grid +
  // DERIVED signs64 planes are emit-period static const decls (NEVER op attrs). ----
  if (loopBody.getFoldModel() == "grid_sign_single_scale_eighth" ||
      loopBody.getFoldModel() == "grid_sign_dualscale_eighth" ||
      loopBody.getFoldModel() == "grid_ternary_delta_eighth" ||
      loopBody.getFoldModel() == "grid_ternary_delta_groupsum_eighth" ||
      loopBody.getFoldModel() == "grid_sign_dual_entry_single_scale_quarter" ||
      // C4a-5: iq3_s. Same grid branch, same dual-entry leaf; its own fold_model only
      // because its STORE constant differs (ggml: `*s = sumf`).
      loopBody.getFoldModel() == "grid_sign_dual_entry_single_scale_unit") {
    weftrvv::RepackGemvGridCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemvGridCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "grid repack GEVM loop body requires the region "
                    "weft_rvv.repack_gemv_grid_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0))
      return rewriter.notifyMatchFailure(
          coreBrick, "the grid core brick's block_index must be the loop "
                     "induction variable (region arg 0)");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the grid core brick's weight/activation bases must be the "
                     "loop-body's own repacked-weight / q8_K-activation ABI "
                     "buffers");
    llvm::StringRef decodeModel = coreBrick.getDecodeModel();
    // The CLOSED weft::GridDecodePlan registry is the SAME fail-closed authority the
    // grid core verifier consults -- an unregistered decode_model has no plan and is
    // REJECTED here ([D-1] unknown = reject), rather than being matched against a
    // hand-synced copy of the verifier's string chain.
    const weft::GridDecodePlan *gridPlan = weft::lookupGridDecodePlan(decodeModel);
    if (!gridPlan)
      return rewriter.notifyMatchFailure(
          coreBrick, "grid repack GEVM decode_model is not registered in the closed "
                     "weft::GridDecodePlan registry");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(
          loopBody, "grid repack GEVM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // The plan's ls ARITY selects the body leaf: Single rides the iq2_xxs leaf, Dual
    // rides the shared dual-ls leaf (which now takes the PLAN, so the grid table + sign
    // plane are DATA rather than an Iq2DualGridVariant hard-select).
    // C4a-2: the plan's FOLD ARITH selects the leaf FIRST -- DeltaGrid (iq1_s) is a
    // structurally different body (no sign gather, TWO integer accumulators, the
    // activation bsums, and NO store-side 0.125), not an ls-arity variant of the iq2
    // leaves. Its bsums offset rides the loop body's OPTIONAL shared attr slot, which
    // the verifier admits for exactly this fold and the q4_K min fold.
    if (gridPlan->foldArith == weft::GridFoldArith::DeltaGrid) {
      std::optional<uint64_t> bsumsOff = loopBody.getActivationBsumsByteOffset();
      if (!bsumsOff)
        return rewriter.notifyMatchFailure(
            loopBody, "iq1_s repack GEVM loop body requires the "
                      "activation_bsums_byte_offset attr (the delta term's "
                      "bsums plane)");
      return emitRepackGemvIq1SQ8K(
          rewriter, loc, *gridPlan, weightBase, activationBase, output,
          columnCount, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          // The sign-plane SLOT carries the +-1 DELTA strip for a TernaryDelta row.
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(*bsumsOff),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    }
    // C4a-3: iq1_m. The SAME "fold arith selects the leaf FIRST" rule -- its ls arity
    // IS Dual, so without this arm it would fall into the iq2 dual-ls leaf and be
    // silently mis-lowered (the iq2 leaf gathers a sign plane iq1_m has no table for,
    // and has ONE accumulator + a store-side 0.125). The delta term is the reason it
    // is not the iq1_s leaf either. Note the DELIBERATE asymmetry with the DeltaGrid
    // arm above: that one REQUIRES the bsums attr; this one demands its ABSENCE, since
    // block_q8_K's per-16 bsums cannot express iq1_m's per-8 delta group sum. The
    // loop-body verifier already rejects the attr for this fold, so the check here is
    // the emitter refusing to lower IR that verifier could not have produced.
    if (gridPlan->foldArith == weft::GridFoldArith::DeltaGridGroupSum) {
      if (loopBody.getActivationBsumsByteOffset())
        return rewriter.notifyMatchFailure(
            loopBody, "iq1_m repack GEVM loop body must NOT carry "
                      "activation_bsums_byte_offset: its per-GROUP-of-8 delta sums "
                      "cannot be expressed by block_q8_K's per-16 bsums plane, so "
                      "the leaf accumulates them in-kernel and reads no bsums");
      return emitRepackGemvIq1MQ8K(
          rewriter, loc, *gridPlan, weightBase, activationBase, output,
          columnCount, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          // The sign-plane SLOT carries the PER-GROUP +-1 DELTA strip.
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    }
    // C4a-4: iq3_xxs. The SAME "decide on the STRUCTURAL key BEFORE ls arity" rule the
    // two arms above encode, one axis over -- and this row is the sharpest case of it.
    // iq3_xxs matches the iq2_xxs leaf on EVERY key that leaf tests: its lsArity IS
    // Single, its foldArith IS SignScaleStore, its grid even has the same 256 entries.
    // The only thing that differs is the entry WIDTH, and it is not cosmetic: a uint32
    // entry carries 4 grid bytes, so an 8-lane group needs TWO of them with the
    // activation range split (ggml: grid1 for q8[j+0], grid2 for q8[j+4]). Without this
    // arm the row would fall through to a leaf that hoists ONE base per group and shifts
    // indices by 3 -- which does not fail, it reads the next entry's bytes for the upper
    // half of every group and returns confident garbage. Byte-exactness of the other five
    // rows is what makes that silence dangerous, so the guard is here rather than in a
    // comment.
    if (gridPlan->entryWidth == weft::GridEntryWidth::I32x4) {
      if (loopBody.getActivationBsumsByteOffset())
        return rewriter.notifyMatchFailure(
            loopBody, "iq3_xxs repack GEVM loop body must NOT carry "
                      "activation_bsums_byte_offset: its fold is the single-accumulator "
                      "SignScaleStore shape and reads no bsums plane");
      return emitRepackGemvGridDualEntryQ8K(
          rewriter, loc, *gridPlan, weightBase, activationBase, output,
          columnCount, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    }
    if (gridPlan->lsArity == weft::GridLsArity::Single)
      return emitRepackGridGemvBodyIq2Xxs(
          rewriter, loc, *gridPlan, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()));
    return emitRepackGemvIq2DualScaleQ8K(
        rewriter, loc, *gridPlan, weightBase, activationBase, output, columnCount,
        avlArg, sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
        static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
        static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
        static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
        static_cast<int64_t>(coreBrick.getNSubblocks()),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()));
  }

  // ---- K-QUANT front-door dispatch (the retired emitRepackGemvQ4KQ8K direct
  // emitter, now CONSTRUCTED through this typed-region front door). When the loop
  // body carries the K-quant fold_model its decomposed in-region brick is the
  // weft_rvv.repack_gemv_kquant_core (decode_model "q4_K", NOT the q4_0 nibble core
  // + dual-fp16 fold). We GATE the emit on that brick's block_index-tied + base-tied
  // anti-bypass, then RE-EMIT the byte-exact q4_K GEVM body (the 8-sub-block 6-bit
  // scale/min lane-wise unpack + the split-32 main dot + the bsums-min correction +
  // the dual d/dmin fp16 fold) from the shared body leaf; byte-exactness to the
  // retired direct emitter is by construction (the SAME emit code). The K-quant
  // super-block decode facts (dmin @32, 6-bit scales/mins @64, activation bsums,
  // n_subblocks) ride on the loop body op's OPTIONAL K-quant attrs. ----
  if (loopBody.getFoldModel() == "kquant_dmin_bsums_min") {
    weftrvv::RepackGemvKQuantCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemvKQuantCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEVM loop body requires the region "
                    "weft_rvv.repack_gemv_kquant_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0))
      return rewriter.notifyMatchFailure(
          coreBrick, "the K-quant core brick's block_index must be the loop "
                     "induction variable (region arg 0)");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the K-quant core brick's weight/activation bases must be the "
                     "loop-body's own repacked-weight / q8_K-activation ABI "
                     "buffers");
    // q4_K (4-bit nibble), q2_K (2-bit), and q5_K (4-bit nibble + qh 5th bit) SHARE the
    // dual d/dmin + bsums-min fold (fold_model "kquant_dmin_bsums_min"); the decode_model
    // WHAT selects the per-family decode leaf. q2_K is the LOWEST-bit min-fold sibling of
    // q4_K; q5_K is q4_K + a qh 5th-bit plane (min-fold WITH qh).
    if (coreBrick.getDecodeModel() != "q4_K" &&
        coreBrick.getDecodeModel() != "q2_K" &&
        coreBrick.getDecodeModel() != "q5_K")
      return rewriter.notifyMatchFailure(
          coreBrick, "K-quant repack GEVM decode_model not recognized (expected "
                     "\"q4_K\", \"q2_K\", or \"q5_K\")");
    // The K-quant super-block decode facts on the loop body op's OPTIONAL attrs
    // (fail-closed, I7): dmin strip byte offset, 6-bit/4-bit scales/mins region byte
    // offset, activation int16 bsums byte offset, and sub-block count.
    std::optional<uint64_t> dminOff = loopBody.getWeightDminByteOffset();
    std::optional<uint64_t> scalesOff = loopBody.getWeightScalesByteOffset();
    std::optional<uint64_t> bsumsOff = loopBody.getActivationBsumsByteOffset();
    std::optional<uint64_t> nSub = loopBody.getNSubblocks();
    if (!dminOff || !scalesOff || !bsumsOff || !nSub)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEVM loop body requires the super-block decode "
                    "attrs weight_dmin_byte_offset / weight_scales_byte_offset / "
                    "activation_bsums_byte_offset / n_subblocks");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEVM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] whole-K-nest schedule axis (the *how*, never
    // the *what*), SHARED across the min-fold K-quant GEVM family (q5_K/q4_K/q2_K): PREFER
    // the explicit main_term_form stamp, else the MEASURED-GATE default -- UNROLLED
    // unless code-volume>budget AND a board measurement records rolled beneficial (measured
    // table EMPTY today => UNROLLED, the byte-exact-neutral shipped form; existing fixtures
    // carry NO stamp => unchanged output). The [ROLL] capability is WIRED but activation is
    // measured-gated (Stage-3 board A/B). GEVM is M=1 (no activation column interleave), so
    // activationInterleave is 1. BYTE-EXACT across both schedules by construction (identical
    // vwmacc accumulation order -- only the dominant per-16-element inner loop is
    // materialized as a runtime emitc.for instead of unrolled).
    mlir::FailureOr<bool> rolled = requireMainTermRolled();
    if (mlir::failed(rolled))
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEVM requires final main_term_form");
    bool rolledMainTerm = *rolled;
    // The q5_K (4-bit nibble + qh 5th bit) sibling shares the SAME leaf signature +
    // facts as q4_K PLUS the qh 5th-bit plane byte offset (weight_qh_byte_offset, the
    // SHARED slot -- here on a MIN fold). Its ONLY delta is the decode leaf (the qh
    // inject onto the q4_K nibble). RE-EMITs the byte-exact q5_K GEVM body.
    if (coreBrick.getDecodeModel() == "q5_K") {
      std::optional<uint64_t> qhOff = loopBody.getWeightQhByteOffset();
      if (!qhOff)
        return rewriter.notifyMatchFailure(
            loopBody, "q5_K repack GEVM loop body requires the "
                      "weight_qh_byte_offset attr (the qh 5th-bit plane)");
      // The whole-K-nest rolledMainTerm schedule (resolved ABOVE, SHARED across the
      // min-fold family) selects the compact rolled inner ii-loop vs the register-resident
      // full unroll. BYTE-EXACT across both by construction.
      return emitRepackKQuantGemvBodyQ5K(
          rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*dminOff), static_cast<int64_t>(*scalesOff),
          static_cast<int64_t>(*bsumsOff), static_cast<int64_t>(*qhOff),
          static_cast<int64_t>(*nSub),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm);
    }
    // The q2_K (2-bit) sibling shares the SAME leaf signature + facts as q4_K (4-bit);
    // the ONLY delta is the decode leaf (2-bit weight peel + 4-bit packed scale/min +
    // single-bsum-per-sub-block). RE-EMITs the byte-exact q2_K GEVM body.
    if (coreBrick.getDecodeModel() == "q2_K")
      return emitRepackKQuantGemvBodyQ2K(
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
          static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm);
    return emitRepackKQuantGemvBodyQ4K(
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
        static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm);
  }

  // ---- K-QUANT q6_K NO-MIN front-door dispatch (the retired emitRepackGemvQ6KQ8K
  // direct emitter, now CONSTRUCTED through this typed-region front door). When the
  // loop body carries the q6_K single-accumulator no-min fold_model its decomposed
  // in-region brick is the weft_rvv.repack_gemv_kquant_core (decode_model "q6_K"). We
  // GATE the emit on that brick's block_index-tied + base-tied anti-bypass, then
  // RE-EMIT the byte-exact q6_K GEVM body (the 6-bit ql|qh two-plane -32 offset-binary
  // weight assembly + the SIGNED int8 scale sign-extend + the SINGLE-accumulator
  // no-min fold) from the shared body leaf. The q6_K super-block decode facts (signed
  // scales @32, qh high-2-bit plane @288, n_subblocks 16 -- NO dmin, NO bsums) ride on
  // the loop body op's OPTIONAL K-quant attrs. ----
  if (loopBody.getFoldModel() == "kquant_single_scale_no_min") {
    weftrvv::RepackGemvKQuantCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemvKQuantCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "q6_K repack GEVM loop body requires the region "
                    "weft_rvv.repack_gemv_kquant_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0))
      return rewriter.notifyMatchFailure(
          coreBrick, "the q6_K core brick's block_index must be the loop "
                     "induction variable (region arg 0)");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the q6_K core brick's weight/activation bases must be the "
                     "loop-body's own repacked-weight / q8_K-activation ABI "
                     "buffers");
    // Both q6_K (6-bit ql|qh two-plane offset-binary) and q3_K (3-bit qs|hmask
    // subtractive) SHARE the SINGLE-accumulator no-min fold ("kquant_single_scale_no_min");
    // the decode_model WHAT selects the per-family decode leaf. q3_K is q6_K's no-min
    // cousin (its hmask SECOND weight plane rides the SHARED weight_qh_byte_offset slot).
    if (coreBrick.getDecodeModel() != "q6_K" &&
        coreBrick.getDecodeModel() != "q3_K")
      return rewriter.notifyMatchFailure(
          coreBrick, "K-quant no-min repack GEVM decode_model not recognized "
                     "(expected \"q6_K\" or \"q3_K\")");
    // The no-min super-block decode facts on the loop body op's OPTIONAL attrs
    // (fail-closed, I7): SIGNED int8 scales region byte offset, the SECOND weight-plane
    // byte offset (q6_K qh high-2-bit / q3_K hmask high-bit), and sub-block count. NO
    // dmin / NO bsums (single-accumulator).
    std::optional<uint64_t> scalesOff = loopBody.getWeightScalesByteOffset();
    std::optional<uint64_t> qhOff = loopBody.getWeightQhByteOffset();
    std::optional<uint64_t> nSub = loopBody.getNSubblocks();
    if (!scalesOff || !qhOff || !nSub)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant no-min repack GEVM loop body requires the super-block "
                    "decode attrs weight_scales_byte_offset / weight_qh_byte_offset / "
                    "n_subblocks");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant no-min repack GEVM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] whole-K-nest schedule axis (the *how*, never
    // the *what*), SHARED across the no-min K-quant GEVM family (q6_K/q3_K): PREFER the
    // explicit main_term_form stamp, else the MEASURED-GATE default -- UNROLLED unless
    // code-volume>budget AND a board measurement records rolled beneficial (measured table
    // EMPTY today => UNROLLED, the byte-exact-neutral shipped form; existing fixtures carry
    // NO stamp => unchanged output). The [ROLL] capability is WIRED but activation is
    // measured-gated (Stage-3 board A/B). GEVM is M=1, so activationInterleave is 1.
    // BYTE-EXACT across both schedules by construction (identical vwmacc accumulation order
    // -- only the dominant inner element loop is materialized as a runtime emitc.for
    // instead of unrolled).
    mlir::FailureOr<bool> rolled = requireMainTermRolled();
    if (mlir::failed(rolled))
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEVM requires final main_term_form");
    bool rolledMainTerm = *rolled;
    // q3_K (3-bit subtractive qs|hmask): the qh slot carries the hmask plane offset.
    // RE-EMITs the byte-exact q3_K GEVM body (the no-min sibling of q6_K).
    if (coreBrick.getDecodeModel() == "q3_K")
      return emitRepackKQuantGemvBodyQ3K(
          rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
          sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*scalesOff), static_cast<int64_t>(*qhOff),
          static_cast<int64_t>(*nSub),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm);
    return emitRepackKQuantGemvBodyQ6K(
        rewriter, loc, weightBase, activationBase, output, columnCount, avlArg,
        sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
        static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
        static_cast<int64_t>(*scalesOff), static_cast<int64_t>(*qhOff),
        static_cast<int64_t>(*nSub),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm);
  }

  // ---- Shape facts (the *how* -- LMUL / strip width -- never the *what*): the
  // per-block strides drive the base advance (loop shape); the within-block byte
  // offsets driving the integer core / scale fold come from the BRICKS (the
  // anti-bypass surface), the SAME division of labor as the flat/super-block loop
  // bodies. numHalves == weight_interleave / half_lanes is the disjoint-strip
  // count (2 @VLEN128, 1 @VLEN256/RVV0.7). ----
  int64_t qk = loopBody.getQk();
  int64_t weightInterleave = loopBody.getWeightInterleave();
  int64_t half = loopBody.getHalfLanes();
  int64_t numHalves = weightInterleave / half;
  int64_t weightStride = loopBody.getWeightBlockStride();
  int64_t activationStride = loopBody.getActivationBlockStride();
  int64_t nibbleBytes = qk / 2;

  // ---- Region walk + region-driven gates (fail-closed, I7). The full-body region
  // carries the (block_index, numHalves per-strip vector acc) entry args, ONE
  // integer-core brick weft_rvv.repack_lane_wise_q4_x_i8_dot producing numHalves
  // per-strip sumi, numHalves per-strip dual-fp16 scale FOLD bricks
  // weft_rvv.repack_dual_fp16_scale_fold (each folding one sumi + one carried acc),
  // and a yield naming the numHalves carried-out vectors. Every brick is
  // block_index-tied (anti-bypass) and dataflow-tied (the strip-h fold consumes the
  // integer brick's strip-h sumi + the strip-h loop-carried acc, the yield names
  // the folds' acc_next), so the emit provably tracks the region's brick +
  // accumulator dataflow, not merely the loop-op attrs. ----
  mlir::Block &coreBlock = loopBody.getBody().front();
  weftrvv::TypedRepackGemvLoopYieldOp yieldOp;
  weftrvv::RepackLaneWiseQ4Q8DotOp coreBrick;
  llvm::SmallVector<weftrvv::RepackDualFp16ScaleFoldOp> foldBricks;
  loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
    if (auto o = llvm::dyn_cast<weftrvv::TypedRepackGemvLoopYieldOp>(bodyOp))
      yieldOp = o;
    else if (auto o = llvm::dyn_cast<weftrvv::RepackLaneWiseQ4Q8DotOp>(bodyOp))
      coreBrick = o;
    else if (auto o = llvm::dyn_cast<weftrvv::RepackDualFp16ScaleFoldOp>(bodyOp))
      foldBricks.push_back(o);
  });
  if (!yieldOp)
    return rewriter.notifyMatchFailure(
        loopBody, "typed repack GEVM loop body requires the loop yield");
  if (static_cast<int64_t>(coreBlock.getNumArguments()) != numHalves + 1)
    return rewriter.notifyMatchFailure(
        loopBody, "typed repack GEVM loop body region must carry the "
                  "(block_index, numHalves per-strip vector acc) entry args");
  mlir::Value blockIndexArg = coreBlock.getArgument(0);

  // The integer CORE brick + its block_index-tied anti-bypass gate: the brick must
  // name the loop induction variable (region arg 0) as its block_index and the
  // loop-body's own weight/activation ABI buffers as its bases (so the emit
  // provably addresses `base + block_index*stride (+ byte_offset)`, never the
  // loop-invariant block 0 or a foreign buffer), and produce ONE per-strip sumi per
  // accumulator strip (numHalves results).
  if (!coreBrick)
    return rewriter.notifyMatchFailure(
        loopBody, "repack GEVM loop body requires the region integer CORE brick "
                  "weft_rvv.repack_lane_wise_q4_x_i8_dot (the per-block lane-wise "
                  "nibble dot)");
  if (coreBrick.getBlockIndex() != blockIndexArg)
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack lane-wise dot brick's block_index must be the loop "
                   "induction variable (region arg 0)");
  if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
      coreBrick.getActivationBase() != loopBody.getActivationBase())
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack lane-wise dot brick's weight/activation bases must "
                   "be the loop-body's own repacked-weight / q8_0-activation ABI "
                   "buffers");
  if (static_cast<int64_t>(coreBrick.getNumResults()) != numHalves)
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack lane-wise dot brick must produce one per-strip sumi "
                   "per accumulator strip (numHalves results)");

  // The numHalves per-strip dual-fp16 scale FOLD bricks, INDEXED by the carried
  // accumulator they consume: the strip-h fold consumes region acc arg (1 + h) AND
  // the integer brick's strip-h sumi (result h), is addressed off the SAME
  // block_index + ABI buffers (anti-bypass), and produces the yield's strip-h
  // acc_next. Every strip shares the ONE within-block scale byte offset (the shared
  // fold leaf applies offset + strip*half*2), so all fold bricks must agree.
  if (static_cast<int64_t>(foldBricks.size()) != numHalves)
    return rewriter.notifyMatchFailure(
        loopBody, "repack GEVM loop body requires one dual-fp16 scale FOLD brick "
                  "weft_rvv.repack_dual_fp16_scale_fold per accumulator strip "
                  "(numHalves bricks)");
  if (static_cast<int64_t>(yieldOp.getAccNext().size()) != numHalves)
    return rewriter.notifyMatchFailure(
        yieldOp, "repack GEVM loop yield must name one carried-out per-strip "
                 "accumulator per strip (numHalves acc_next)");
  llvm::SmallVector<weftrvv::RepackDualFp16ScaleFoldOp> foldByStrip(numHalves);
  for (int64_t h = 0; h < numHalves; ++h) {
    mlir::Value accArg = coreBlock.getArgument(1 + h);
    weftrvv::RepackDualFp16ScaleFoldOp fb;
    for (weftrvv::RepackDualFp16ScaleFoldOp cand : foldBricks) {
      if (cand.getAcc() == accArg) {
        fb = cand;
        break;
      }
    }
    if (!fb)
      return rewriter.notifyMatchFailure(
          loopBody, "each per-strip accumulator region argument must be consumed "
                    "by exactly one dual-fp16 scale FOLD brick");
    if (fb.getBlockIndex() != blockIndexArg)
      return rewriter.notifyMatchFailure(
          fb, "the repack scale-fold brick's block_index must be the loop "
              "induction variable (region arg 0)");
    if (fb.getWeightBase() != loopBody.getWeightBase() ||
        fb.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          fb, "the repack scale-fold brick's weight/activation bases must be the "
              "loop-body's own repacked-weight / q8_0-activation ABI buffers");
    if (fb.getSumi() != coreBrick.getResults()[h])
      return rewriter.notifyMatchFailure(
          fb, "the repack scale-fold brick for strip h must consume the integer "
              "CORE brick's strip-h per-strip sumi (result h)");
    if (yieldOp.getAccNext()[h] != fb.getAccNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "the repack GEVM loop yield's strip-h acc_next must be the "
                   "strip-h scale-fold brick's folded-out accumulator");
    if (fb.getWeightScaleByteOffset() !=
            foldBricks.front().getWeightScaleByteOffset() ||
        fb.getActivationScaleByteOffset() !=
            foldBricks.front().getActivationScaleByteOffset())
      return rewriter.notifyMatchFailure(
          fb, "all per-strip scale-fold bricks must share the ONE within-block "
              "fp16 scale byte offset");
    foldByStrip[h] = fb;
  }

  // ---- ABI operands (the same five the monolithic repack GEVM reads). ----
  mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
  mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
  mlir::Value output = valueMap.lookup(loopBody.getOutput());
  mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
  if (!weightBase || !activationBase || !output || !columnCount)
    return rewriter.notifyMatchFailure(loopBody,
                                       "repack GEVM loop ABI operand unmapped");

  llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();

  int64_t weightQuantOffset = coreBrick.getWeightQuantByteOffset();
  int64_t activationQuantOffset = coreBrick.getActivationQuantByteOffset();
  int64_t weightScaleOffset = foldByStrip[0].getWeightScaleByteOffset();
  int64_t activationScaleOffset = foldByStrip[0].getActivationScaleByteOffset();
  // q4_1 decode-leaf facts sourced from the region BRICKS (the anti-bypass
  // surface): the UNSIGNED-nibble flag from the integer CORE brick, and the single
  // MIN-fold offset pair from the dual-fp16 FOLD bricks (all fold bricks agree; -1
  // sentinel = the q4_0 no-min fold, byte-identical).
  bool unsignedNibble = coreBrick.getWeightNibbleUnsigned();
  // q8_0 FULL-int8 decode selector sourced from the CORE brick (the anti-bypass
  // decode-leaf surface): the SIMPLEST flat core (no nibble unpack, i32 in-block
  // accumulation over qk positions).
  bool coreFullI8 = coreBrick.getWeightFullI8();
  // q5_0 5th-bit (qh) decode facts sourced from the CORE brick (the anti-bypass
  // decode-leaf surface): the transposed qh SECOND weight-plane byte offset + the
  // offset-binary centering bias, PRESENT together only for q5_0.
  bool coreHasQh = coreBrick.getWeightQhByteOffset().has_value();
  int64_t coreWeightQhOffset =
      coreHasQh ? static_cast<int64_t>(*coreBrick.getWeightQhByteOffset()) : 0;
  int64_t coreOffsetBias =
      coreBrick.getWeightOffsetBias().has_value()
          ? static_cast<int64_t>(*coreBrick.getWeightOffsetBias())
          : 0;
  int64_t weightMinOffset =
      foldByStrip[0].getWeightMinByteOffset().has_value()
          ? static_cast<int64_t>(*foldByStrip[0].getWeightMinByteOffset())
          : -1;
  int64_t activationSumOffset =
      foldByStrip[0].getActivationSumByteOffset().has_value()
          ? static_cast<int64_t>(*foldByStrip[0].getActivationSumByteOffset())
          : -1;

  // The integer-core LMUL anchor (the *how*, never the *what*): "mf2" (the RVV1.0
  // fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2, f16 scale m1) or
  // "m1" (the RVV0.7 whole-LMUL chain i8m1 -> i16m2 -> i32m4 -> f32m4, f16 scale
  // m2). Only the type/callee LMUL suffixes change; numHalves, vl, every loop bound
  // and byte offset are driven by half_lanes and stay identical -- exactly the
  // monolithic emitRepackGemvQ4_0Q8_0's rung derivation.
  // Fail-closed final capability-fact read: the
  // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
  if (!loopBody.getIntegerCoreLmul())
    return rewriter.notifyMatchFailure(
        loopBody, "repack integer core requires an explicit integer_core_lmul "
              "capability fact (front door stamps it; no silent mf2 default)");
  llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
  llvm::StringRef l8 = coreLmul;                         // mf2 -> mf2; m1 -> m1
  llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";  // mf2 -> m1;  m1 -> m2
  llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";  // mf2 -> m2;  m1 -> m4
  mlir::Type f32AccType =
      emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type floatPtrType = output.getType();
  mlir::Type weightPtrType = weightBase.getType();
  mlir::Type activationPtrType = activationBase.getType();

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto step = [&](llvm::StringRef s) {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // The active vl is the compile-time-constant strip width (half_lanes e16m1 lanes:
  // 16 for the one-strip VLEN=256/RVV0.7 form, 8 for the two-halves VLEN=128 form),
  // exactly as the monolithic repack GEVM runs every intrinsic.
  mlir::Value vl = sizeLit(half);

  // size_t nb = n / QK;  (the contraction block count).
  step("block_count");
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
  // size_t nc_groups = nc / 16;
  step("col_group_count");
  mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
      loc, sizeType, columnCount, sizeLit(weightInterleave));

  // ===== Outer weight-COLUMN-GROUP loop: for (x = 0; x < nc/16; ++x) =====
  auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncGroups,
                                               sizeLit(1),
                                               /*bodyBuilder=*/nullptr);
  {
    mlir::OpBuilder::InsertionGuard cg(rewriter);
    rewriter.setInsertionPointToStart(colLoop.getBody());
    mlir::Value x = colLoop.getInductionVar();

    // const uint8_t *b = vx + x*nb*288;  (the q4_0x16 column group base).
    // Byte-exact to the monolithic bGroup; the per-block base advances off it.
    step("weight_group_base");
    mlir::Value bGroupBlocks =
        rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
    mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
        loc, sizeType, bGroupBlocks, sizeLit(weightStride));
    mlir::Value bGroup = rewriter.create<emitc::AddOp>(
        loc, weightPtrType, weightBase, bGroupOff);

    // vfloat32{m2,m4}_t sumf_h = vfmv_v_f(0.0f, vl);  -- ONE per-strip f32 vector
    // accumulator per strip (byte-exact to the monolithic seedF32; numHalves at
    // half=8, one at half=16). Each SSA loop-carried `acc` block argument (region
    // arg 1 + h) lowers to its mutable emitc.variable lvalue.
    std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
    llvm::SmallVector<mlir::Value> sumfVar;
    for (int64_t h = 0; h < numHalves; ++h) {
      auto v = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(f32AccType),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, f32AccType, fmvCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                    .getResult();
            return {zero, vl};
          });
      rewriter.create<emitc::AssignOp>(loc, v, seed);
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

      // const uint8_t *bl = b + l*288;   const uint8_t *al = a + l*34;  The
      // per-block bases advance off the loop induction variable `l` (== the
      // bricks' gated block_index) by the loop-op strides -- the block_index-tied
      // anti-bypass addressing, byte-exact to the monolithic bl/al.
      step("weight_block_base");
      mlir::Value blOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, l, sizeLit(weightStride));
      mlir::Value bl =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, bGroup, blOff);
      step("act_block_base");
      mlir::Value alOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, l, sizeLit(activationStride));
      mlir::Value al = rewriter.create<emitc::AddOp>(loc, activationPtrType,
                                                     activationBase, alOff);

      // ===== The region integer CORE: the weft_rvv.repack_lane_wise_q4_x_i8_dot
      // brick lowers to the SHARED emitRepackQ4LaneWiseIntegerCore leaf (per strip:
      // seed i16 lo/hi -> nibble-step vwmacc loop -> lo/hi vwadd combine), byte-
      // exact to the monolithic emitRepackGemvQ4_0Q8_0's integer part, producing
      // the numHalves per-strip sumi. The within-block byte offsets are sourced
      // from the BRICK (the anti-bypass surface -- a rewired offset changes the
      // emitted addresses). =====
      RepackQ4IntegerCoreContext coreCx{
          opName, role, l8, l16, l32, numHalves, half, nibbleBytes,
          weightInterleave, weightQuantOffset, activationQuantOffset,
          nibbleBytes, vl, sizeType};
      coreCx.unsignedNibble = unsignedNibble;
      coreCx.hasQh = coreHasQh;
      coreCx.weightQhByteOffset = coreWeightQhOffset;
      coreCx.offsetBias = coreOffsetBias;
      coreCx.fullI8 = coreFullI8;
      llvm::SmallVector<mlir::Value> sumi =
          emitRepackQ4LaneWiseIntegerCore(rewriter, loc, coreCx, bl, al);

      // ===== The region dual-fp16 scale FOLD: the numHalves
      // weft_rvv.repack_dual_fp16_scale_fold bricks lower through ONE call to the
      // SHARED emitRepackDualFp16ScaleFold leaf (per strip: vle16 weight scale ->
      // ONE _Float16 act scale -> vfwmul/vfcvt/vfmacc into the strip's f32
      // accumulator; load sumf_h, fold, assign back), byte-exact to the monolithic
      // emitRepackGemvQ4_0Q8_0's fold part. The within-block scale byte offsets are
      // sourced from the FOLD BRICKS (the anti-bypass surface -- a rewired offset
      // changes the emitted scale addresses). The carried-OUT accumulators are the
      // fold results (the yield's acc_next). =====
      RepackDualFp16ScaleFoldContext foldCx{
          opName, role, l16, l32, numHalves, half, weightScaleOffset,
          activationScaleOffset, vl, sizeType};
      foldCx.weightMinByteOffset = weightMinOffset;
      foldCx.activationSumByteOffset = activationSumOffset;
      emitRepackDualFp16ScaleFold(rewriter, loc, foldCx, bl, al, sumi, sumfVar);
    }

    // Per-strip lane-wise vector store vse32(s + x*16 + h*half, sumf_h, vl) (NO
    // horizontal reduction -- each repacked block-as-lane strip writes its columns
    // straight to s + x*16 + h*half). Byte-exact to the monolithic storeHalf; the
    // h==0 first store is AddOp-free.
    std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
    for (int64_t h = 0; h < numHalves; ++h) {
      step("output_addr");
      mlir::Value x16 = rewriter.create<emitc::MulOp>(
          loc, sizeType, x, sizeLit(weightInterleave));
      mlir::Value totalOff = x16;
      if (h * half != 0)
        totalOff = rewriter.create<emitc::AddOp>(loc, sizeType, x16,
                                                 sizeLit(h * half));
      mlir::Value dst = rewriter.create<emitc::AddOp>(loc, floatPtrType, output,
                                                      totalOff);
      mlir::Value sumfVal =
          rewriter.create<emitc::LoadOp>(loc, f32AccType, sumfVar[h])
              .getResult();
      emitOpaqueCallVoid(rewriter, loc, vseCallee,
                         mlir::ValueRange{dst, sumfVal, vl}, opName, role);
    }
  }

  return mlir::success();
}

// M-FLAT REPACK GEMM finale M2: the typed region-carrying sibling of the
// monolithic emitRepackGemmQ4_0Q8_0. It lowers weft_rvv.typed_repack_gemm_loop_body
// -- whose region carries the inner contraction-block loop with the columnsPerPass
// per-column LANE-WISE f32 VECTOR loop-carried accumulators (plus the block_index
// and the runtime strip_row_offset entry args) -- to the byte-exact repacked GEMM
// kernel by wrapping that block loop in the SAME four outer loops the monolith
// emits: the activation-ROW-group (M-tiling) loop over nr/activation_interleave,
// the weight-column-group loop over nc/weight_interleave, the RUNTIME strip loop
// over numHalves (roff = h*half_lanes), and the compile-time column-PASS loop
// (columnsPerPass of the activation_interleave interleaved columns per pass). The
// inner body drives the SHARED GEMM leaves emitRepackGemmQ4LaneWiseIntegerCore (the
// ONE-strip N-column integer core, producing columnsPerPass per-column sumi) +
// emitRepackGemmDualFp16ScaleFold (the per-column dual-fp16 scale fold) -- the SAME
// leaves the monolith calls -- from the region's CORE + FOLD bricks, so the whole
// kernel body is byte-identical to emitRepackGemmQ4_0Q8_0 by construction on every
// arm (the monolith's trailing unused-result token is its only residue). The emit
// GATES on the core brick's block_index + strip_row_offset anti-bypass ties and its
// weight/activation ABI bases, on EACH fold brick's block_index + strip + base +
// sumi(result c) + acc(region arg 2+c) dataflow tie, and on the yield naming the
// folds' acc_next; the within-block weight/activation quant byte offsets driving the
// integer core are SOURCED from the CORE brick (the anti-bypass surface).

mlir::LogicalResult VariantToEmitCFunc::emitTypedRepackGemmLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::MLIRContext *ctx = rewriter.getContext();

  weftrvv::TypedRepackGemmLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<weftrvv::TypedRepackGemmLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed repack GEMM loop body missing the op");

  auto requireMainTermRolled = [&]() -> mlir::FailureOr<bool> {
    std::optional<llvm::StringRef> form = loopBody.getMainTermForm();
    if (!form || (*form != "unrolled" && *form != "rolled"))
      return mlir::failure();
    return *form == "rolled";
  };

  // The loop order is a required final typed field. Construction has already
  // performed candidate generation, legality, prior and optional winner lookup;
  // emission only realizes the selected nest.
  const bool selectedColGroupOuter = loopBody.getLoopOrder() == "col_outer";

  // ---- TERNARY front-door dispatch (the retired emitRepackGem{m}TQ{20,10}Q8K
  // direct emitters, now CONSTRUCTED through this typed-region front door). Gate on
  // the in-region weft_rvv.repack_gemm_ternary_core brick's block_index +
  // strip_row_offset + base anti-bypass ties, then RE-EMIT the byte-exact ternary
  // GEMM body from the shared body leaf. ----
  if (loopBody.getFoldModel() == "ternary_single_fp16_scale") {
    weftrvv::RepackGemmTernaryCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemmTernaryCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "ternary repack GEMM loop body requires the region "
                    "weft_rvv.repack_gemm_ternary_core integer-core brick");
    if (coreBrick.getBlockIndex() !=
            loopBody.getBody().front().getArgument(0) ||
        coreBrick.getStripRowOffset() !=
            loopBody.getBody().front().getArgument(1))
      return rewriter.notifyMatchFailure(
          coreBrick, "the ternary GEMM core brick's block_index / "
                     "strip_row_offset must be region args 0 / 1");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the ternary GEMM core brick's weight/activation bases must "
                     "be the loop-body's own repacked-weight / q8_Kx4-activation "
                     "ABI buffers");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value rowCount = valueMap.lookup(loopBody.getRowCount());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    mlir::Value outputRowStride =
        valueMap.lookup(loopBody.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(
          loopBody, "ternary repack GEMM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    if (coreBrick.getDecodeModel() == "tq2_0")
      return emitRepackTernaryGemmBodyTQ20(
          rewriter, loc, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    // tq1_0 base-3: the base-3 decode reads a SECOND weight plane (qh), whose
    // repacked byte offset rides on the loop body op's OPTIONAL weight_qh_byte_offset
    // attr (the tq2_0 single-plane fold does not carry it).
    if (coreBrick.getDecodeModel() == "tq1_0") {
      std::optional<uint64_t> qhOff = loopBody.getWeightQhByteOffset();
      if (!qhOff)
        return rewriter.notifyMatchFailure(
            loopBody, "tq1_0 ternary repack GEMM loop body requires the base-3 "
                      "weight_qh_byte_offset SECOND-plane attr");
      return emitRepackTernaryGemmBodyTQ10(
          rewriter, loc, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(*qhOff),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    }
    return rewriter.notifyMatchFailure(
        coreBrick, "ternary repack GEMM decode_model not recognized (expected "
                   "\"tq2_0\" 2-bit or \"tq1_0\" base-3)");
  }

  // ---- CODEBOOK front-door dispatch (the retired emitRepackGem{m}Iq4{Nl,Xs} direct
  // emitters, now CONSTRUCTED through this typed-region front door). Gate on the in-region
  // weft_rvv.repack_gemm_codebook_core brick's block_index + strip_row_offset + base
  // anti-bypass ties, then RE-EMIT the byte-exact codebook GEMM body (iq4_nl flat OR
  // iq4_xs super-block signed-6) from the shared body leaf. The GEMM body ships PLAIN
  // (untiled): iq4_nl/iq4_xs already sit at the <=32-vreg cliff, so S6 output tiling is a
  // structural no-op. The 16-entry non-linear codebook rides the core brick's
  // DenseI8ArrayAttr (the REAL vluxei16 gather WHAT). ----
  bool isCodebookFlatFoldM =
      loopBody.getFoldModel() == "codebook_flat_single_scale";
  bool isCodebookSuperblockFoldM =
      loopBody.getFoldModel() == "codebook_superblock_signed6_no_min";
  // The mxfp4 FLAT codebook fold WITH the E8M0 shared-exponent per-column scale
  // (decode_model "mxfp4"; the GEMM sibling of the GEVM E8M0 fold).
  bool isCodebookE8m0FoldM =
      loopBody.getFoldModel() == "codebook_flat_e8m0_scale";
  if (isCodebookFlatFoldM || isCodebookSuperblockFoldM || isCodebookE8m0FoldM) {
    weftrvv::RepackGemmCodebookCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemmCodebookCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "codebook repack GEMM loop body requires the region "
                    "weft_rvv.repack_gemm_codebook_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0) ||
        coreBrick.getStripRowOffset() !=
            loopBody.getBody().front().getArgument(1))
      return rewriter.notifyMatchFailure(
          coreBrick, "the codebook GEMM core brick's block_index / "
                     "strip_row_offset must be region args 0 / 1");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the codebook GEMM core brick's weight/activation bases must "
                     "be the loop-body's own repacked-weight / q8x4-activation "
                     "ABI buffers");
    // The decode_model MUST agree with the fold_model: iq4_nl on the flat fold,
    // iq4_xs on the super-block signed-6 fold, mxfp4 on the flat E8M0 fold
    // (fail-closed, I7).
    bool isIq4Xs = coreBrick.getDecodeModel() == "iq4_xs";
    bool isIq4Nl = coreBrick.getDecodeModel() == "iq4_nl";
    bool isMxfp4 = coreBrick.getDecodeModel() == "mxfp4";
    if ((isCodebookSuperblockFoldM && !isIq4Xs) ||
        (isCodebookFlatFoldM && !isIq4Nl) ||
        (isCodebookE8m0FoldM && !isMxfp4))
      return rewriter.notifyMatchFailure(
          coreBrick, "codebook repack GEMM decode_model not recognized / does not "
                     "match the fold_model (expected \"iq4_nl\" on the flat fold, "
                     "\"iq4_xs\" on the super-block signed-6 fold, or \"mxfp4\" on "
                     "the flat E8M0 fold)");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value rowCount = valueMap.lookup(loopBody.getRowCount());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(loopBody.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(
          loopBody, "codebook repack GEMM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // The iq4_xs SUPER-BLOCK sibling: the SAME codebook gather + i32 dot PLUS the
    // K-quant 6-bit SIGNED per-sub-block scale fold, AMORTIZED across the 4 interleaved
    // block_q8_Kx4 columns. Its super-block decode facts ride on the loop body op's
    // OPTIONAL attrs (fail-closed, I7).
    if (isIq4Xs) {
      std::optional<uint64_t> scalesLow = loopBody.getWeightScalesByteOffset();
      std::optional<uint64_t> scalesHigh =
          loopBody.getWeightScalesHighByteOffset();
      std::optional<uint64_t> nSub = loopBody.getNSubblocks();
      if (!scalesLow || !scalesHigh || !nSub)
        return rewriter.notifyMatchFailure(
            loopBody, "iq4_xs codebook repack GEMM loop body requires the "
                      "super-block decode attrs weight_scales_byte_offset / "
                      "weight_scales_high_byte_offset / n_subblocks");
      return emitRepackCodebookGemmBodyIq4Xs(
          rewriter, loc, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(*scalesLow), static_cast<int64_t>(*scalesHigh),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*nSub), coreBrick.getCodebook(),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    }
    // The mxfp4 FLAT E8M0 sibling: the SAME codebook gather + i32 dot AMORTIZED across the
    // 4 interleaved block_q8_0x4 columns, but the per-column weight scale is the E8M0
    // shared-exponent bit-construction (NO fp16 scale, NO min, NO sub-block). Same body-leaf
    // signature as iq4_nl (flat, nSubblocks == 0).
    if (isMxfp4)
      return emitRepackCodebookGemmBodyMxfp4(
          rewriter, loc, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          coreBrick.getCodebook(),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    return emitRepackCodebookGemmBodyIq4Nl(
        rewriter, loc, weightBase, activationBase, output, rowCount, columnCount,
        outputRowStride, avlArg, sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
        static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
        coreBrick.getCodebook(),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getActivationInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
  }

  // ---- GRID front-door dispatch (the retired emitRepackGemmIq2XxsQ8K direct emitter,
  // now CONSTRUCTED through this typed-region front door). Gate on the in-region
  // weft_rvv.repack_gemm_grid_core brick's block_index + strip_row_offset + base
  // anti-bypass ties, then RE-EMIT the byte-exact iq2_xxs GEMM body (the SAME grid GATHER
  // + sign-plane GATHER + ls-scale fold AMORTIZED across the 4 interleaved block_q8_Kx4
  // columns) from the shared body leaf; byte-exact to the retired direct emitter by
  // construction. The grid decode facts ride on the grid core brick; the fixed grid +
  // signs64 planes are DERIVED static const decls. iq2_xxs ships PLAIN (untiled): a
  // memory-gather grid decode already <= the 32-vreg cliff. ----
  if (loopBody.getFoldModel() == "grid_sign_single_scale_eighth" ||
      loopBody.getFoldModel() == "grid_sign_dualscale_eighth" ||
      loopBody.getFoldModel() == "grid_ternary_delta_eighth" ||
      loopBody.getFoldModel() == "grid_ternary_delta_groupsum_eighth" ||
      loopBody.getFoldModel() == "grid_sign_dual_entry_single_scale_quarter" ||
      // C4a-5: iq3_s. Same grid branch, same dual-entry leaf; its own fold_model only
      // because its STORE constant differs (ggml: `*s = sumf`).
      loopBody.getFoldModel() == "grid_sign_dual_entry_single_scale_unit") {
    weftrvv::RepackGemmGridCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemmGridCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "grid repack GEMM loop body requires the region "
                    "weft_rvv.repack_gemm_grid_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0))
      return rewriter.notifyMatchFailure(
          coreBrick, "the grid core brick's block_index must be the loop "
                     "induction variable (region arg 0)");
    if (coreBrick.getStripRowOffset() != loopBody.getBody().front().getArgument(1))
      return rewriter.notifyMatchFailure(
          coreBrick, "the grid core brick's strip_row_offset must be the region "
                     "strip offset (region arg 1)");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the grid core brick's weight/activation bases must be the "
                     "loop-body's own repacked-weight / q8_K-activation ABI "
                     "buffers");
    llvm::StringRef decodeModel = coreBrick.getDecodeModel();
    // The CLOSED weft::GridDecodePlan registry -- the SAME fail-closed authority the
    // grid core verifier and the repack GEVM leaf consult ([D-1] unknown = reject).
    const weft::GridDecodePlan *gridPlan = weft::lookupGridDecodePlan(decodeModel);
    if (!gridPlan)
      return rewriter.notifyMatchFailure(
          coreBrick, "grid repack GEMM decode_model is not registered in the closed "
                     "weft::GridDecodePlan registry");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value rowCount = valueMap.lookup(loopBody.getRowCount());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(loopBody.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(
          loopBody, "grid repack GEMM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // The plan's ls ARITY selects the body leaf: Single rides the iq2_xxs leaf, Dual
    // rides the shared dual-ls leaf (which now takes the PLAN, so the grid table + sign
    // plane are DATA rather than an Iq2DualGridVariant hard-select).
    // C4a-2: FOLD ARITH first (see the GEVM sibling) -- DeltaGrid is a different
    // body, not an ls-arity variant.
    if (gridPlan->foldArith == weft::GridFoldArith::DeltaGrid) {
      std::optional<uint64_t> bsumsOff = loopBody.getActivationBsumsByteOffset();
      if (!bsumsOff)
        return rewriter.notifyMatchFailure(
            loopBody, "iq1_s repack GEMM loop body requires the "
                      "activation_bsums_byte_offset attr (the delta term's "
                      "bsums plane)");
      return emitRepackGemmIq1SQ8K(
          rewriter, loc, *gridPlan, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          // The sign-plane SLOT carries the +-1 DELTA strip for a TernaryDelta row.
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(*bsumsOff),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    }
    // C4a-3: iq1_m (the GEMM sibling -- see the GEVM arm for why fold arith must be
    // consulted BEFORE ls arity: iq1_m's arity IS Dual, so without this arm it would
    // be silently mis-lowered by the iq2 dual-ls leaf).
    if (gridPlan->foldArith == weft::GridFoldArith::DeltaGridGroupSum) {
      if (loopBody.getActivationBsumsByteOffset())
        return rewriter.notifyMatchFailure(
            loopBody, "iq1_m repack GEMM loop body must NOT carry "
                      "activation_bsums_byte_offset: its per-GROUP-of-8 delta sums "
                      "cannot be expressed by block_q8_K's per-16 bsums plane, so "
                      "the leaf accumulates them in-kernel and reads no bsums");
      return emitRepackGemmIq1MQ8K(
          rewriter, loc, *gridPlan, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          // The sign-plane SLOT carries the PER-GROUP +-1 DELTA strip.
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    }
    // C4a-4: iq3_xxs, the prefill half of the GEVM arm's argument -- entryWidth is tested
    // BEFORE ls arity because iq3_xxs is Single-ls and would otherwise be silently
    // mis-lowered onto the single-base iq2_xxs leaf. See the GEVM dispatcher above and
    // GridDecodePlan.h's leaf-selection order.
    if (gridPlan->entryWidth == weft::GridEntryWidth::I32x4) {
      if (loopBody.getActivationBsumsByteOffset())
        return rewriter.notifyMatchFailure(
            loopBody, "iq3_xxs repack GEMM loop body must NOT carry "
                      "activation_bsums_byte_offset: its fold is the single-accumulator "
                      "SignScaleStore shape and reads no bsums plane");
      return emitRepackGemmGridDualEntryQ8K(
          rewriter, loc, *gridPlan, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    }
    if (gridPlan->lsArity == weft::GridLsArity::Single)
      return emitRepackGridGemmBodyIq2Xxs(
          rewriter, loc, *gridPlan, weightBase, activationBase, output, rowCount,
          columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
          static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
          static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
          static_cast<int64_t>(coreBrick.getNSubblocks()),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
    return emitRepackGemmIq2DualScaleQ8K(
        rewriter, loc, *gridPlan, weightBase, activationBase, output, rowCount,
        columnCount, outputRowStride, avlArg, sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(coreBrick.getWeightQuantByteOffset()),
        static_cast<int64_t>(coreBrick.getWeightLsByteOffset()),
        static_cast<int64_t>(coreBrick.getWeightSignByteOffset()),
        static_cast<int64_t>(coreBrick.getActivationQuantByteOffset()),
        static_cast<int64_t>(coreBrick.getNSubblocks()),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getActivationInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()), selectedColGroupOuter);
  }

  // ---- K-QUANT front-door dispatch (the retired emitRepackGemmQ4KQ8K direct
  // emitter, now CONSTRUCTED through this typed-region front door). Gate on the
  // in-region weft_rvv.repack_gemm_kquant_core brick's block_index + strip_row_offset
  // + base anti-bypass ties, then RE-EMIT the byte-exact q4_K GEMM body from the
  // shared body leaf. ----
  if (loopBody.getFoldModel() == "kquant_dmin_bsums_min") {
    weftrvv::RepackGemmKQuantCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemmKQuantCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEMM loop body requires the region "
                    "weft_rvv.repack_gemm_kquant_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0) ||
        coreBrick.getStripRowOffset() !=
            loopBody.getBody().front().getArgument(1))
      return rewriter.notifyMatchFailure(
          coreBrick, "the K-quant GEMM core brick's block_index / "
                     "strip_row_offset must be region args 0 / 1");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the K-quant GEMM core brick's weight/activation bases must "
                     "be the loop-body's own repacked-weight / q8_Kx4-activation "
                     "ABI buffers");
    // q4_K (4-bit), q2_K (2-bit), and q5_K (4-bit nibble + qh 5th bit) SHARE the dual
    // d/dmin + bsums-min GEMM fold; the decode_model WHAT selects the per-family decode
    // leaf. All three leaves are S6-TILED (the register-cliff lever transfers -- the
    // min+bsums fold is the same shape; q5_K's qh inject is orthogonal to the tiling).
    if (coreBrick.getDecodeModel() != "q4_K" &&
        coreBrick.getDecodeModel() != "q2_K" &&
        coreBrick.getDecodeModel() != "q5_K")
      return rewriter.notifyMatchFailure(
          coreBrick, "K-quant repack GEMM decode_model not recognized (expected "
                     "\"q4_K\", \"q2_K\", or \"q5_K\")");
    std::optional<uint64_t> dminOff = loopBody.getWeightDminByteOffset();
    std::optional<uint64_t> scalesOff = loopBody.getWeightScalesByteOffset();
    std::optional<uint64_t> bsumsOff = loopBody.getActivationBsumsByteOffset();
    std::optional<uint64_t> nSub = loopBody.getNSubblocks();
    if (!dminOff || !scalesOff || !bsumsOff || !nSub)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEMM loop body requires the super-block decode "
                    "attrs weight_dmin_byte_offset / weight_scales_byte_offset / "
                    "activation_bsums_byte_offset / n_subblocks");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value rowCount = valueMap.lookup(loopBody.getRowCount());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(loopBody.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEMM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // [GAP-EMIT-VSETVL-TAX] / [K-10] structural GEMM plan: the whole-K-nest [ROLL]
    // schedule axis (the *how*, never the *what*), resolved ONCE for the WHOLE min-fold
    // K-quant GEMM family (q4_K/q2_K/q5_K share it -- the discriminant is the code-volume
    // vs I-cache budget FACT, NEVER the format spelling). PREFER the explicit front-door
    // main_term_form stamp, else the MEASURED-GATE default -- UNROLLED unless
    // code-volume>budget AND a board measurement records rolled beneficial (measured table
    // EMPTY today => UNROLLED; the [ROLL] capability is WIRED but activation is
    // measured-gated, Stage-3 board A/B). ORTHOGONAL to the q4_K colGroupOuter loop-order
    // axis (both coexist). BYTE-EXACT across both schedules by construction (identical
    // vwmacc16 accumulation order; only the loop is materialized).
    mlir::FailureOr<bool> rolled = requireMainTermRolled();
    if (mlir::failed(rolled))
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEMM requires final main_term_form");
    bool rolledMainTerm = *rolled;
    // q5_K (4-bit nibble + qh 5th bit): the S6-tiled q4_K GEMM body WITH the qh inject.
    // Requires the weight_qh_byte_offset attr (the qh 5th-bit plane, the SHARED slot on a
    // MIN fold). RE-EMITs the byte-exact S6-tiled q5_K GEMM body.
    if (coreBrick.getDecodeModel() == "q5_K") {
      std::optional<uint64_t> qhOff = loopBody.getWeightQhByteOffset();
      if (!qhOff)
        return rewriter.notifyMatchFailure(
            loopBody, "q5_K repack GEMM loop body requires the "
                      "weight_qh_byte_offset attr (the qh 5th-bit plane)");
      return emitRepackKQuantGemmBodyQ5K(
          rewriter, loc, weightBase, activationBase, output, rowCount, columnCount,
          outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*dminOff), static_cast<int64_t>(*scalesOff),
          static_cast<int64_t>(*bsumsOff), static_cast<int64_t>(*qhOff),
          static_cast<int64_t>(*nSub),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm,
          selectedColGroupOuter);
    }
    if (coreBrick.getDecodeModel() == "q2_K") {
      // [ROLL] schedule resolved ABOVE for the whole min-fold family (rolledMainTerm).
      return emitRepackKQuantGemmBodyQ2K(
          rewriter, loc, weightBase, activationBase, output, rowCount, columnCount,
          outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*dminOff), static_cast<int64_t>(*scalesOff),
          static_cast<int64_t>(*bsumsOff), static_cast<int64_t>(*nSub),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm,
          selectedColGroupOuter);
    }
    // q4_K consumes the same selected loop-order plan as every sibling. There is
    // no emitter-local stride prior and no reason gate.
    return emitRepackKQuantGemmBodyQ4K(
        rewriter, loc, weightBase, activationBase, output, rowCount, columnCount,
        outputRowStride, avlArg, sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
        static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
        static_cast<int64_t>(*dminOff), static_cast<int64_t>(*scalesOff),
        static_cast<int64_t>(*bsumsOff), static_cast<int64_t>(*nSub),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getActivationInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm,
        selectedColGroupOuter);
  }

  // ---- K-QUANT q6_K NO-MIN front-door dispatch (the retired emitRepackGemmQ6KQ8K
  // direct emitter, now CONSTRUCTED through this typed-region front door). Gate on the
  // in-region weft_rvv.repack_gemm_kquant_core brick's block_index + strip_row_offset +
  // base anti-bypass ties (decode_model "q6_K"), then RE-EMIT the byte-exact S6-tiled
  // q6_K GEMM body (block-top f32 d widen + per-block scaleVal, NO min-fold) from the
  // shared body leaf. [G8 stage-3, board-remeasured 2026-07-14] The S6 STRIP-OUTER
  // register-cliff lever DOES transfer to the no-min q6_K/q3_K family (the earlier
  // "family-lever NULL / q4_K's lever does not transfer" claim was FALSIFIED on
  // rvv/VLEN128 cold A/B): the strip-INTERLEAVED body spilled ~2068 whole-reg (vs1r/vl1r)
  // + ~2210 csrr vlenb slots because ALL 2*numHalves*columnsPerPass accumulators are
  // live across ONE block loop (~64 vreg > 32-budget). Hoisting the weight-strip loop
  // OUTSIDE the block loop halves the live set (~36 vreg): objdump total 20960->9215,
  // spill ~22x fewer; q6_K prefill cold ratio 0.15->0.96(nr16)/1.14(nr64) WIN. Byte-exact
  // (S6 == the old strip-inner emit, 27/27 shapes). ----
  if (loopBody.getFoldModel() == "kquant_single_scale_no_min") {
    weftrvv::RepackGemmKQuantCoreOp coreBrick;
    loopBody.getBody().walk(
        [&](weftrvv::RepackGemmKQuantCoreOp o) { coreBrick = o; });
    if (!coreBrick)
      return rewriter.notifyMatchFailure(
          loopBody, "q6_K repack GEMM loop body requires the region "
                    "weft_rvv.repack_gemm_kquant_core integer-core brick");
    if (coreBrick.getBlockIndex() != loopBody.getBody().front().getArgument(0) ||
        coreBrick.getStripRowOffset() !=
            loopBody.getBody().front().getArgument(1))
      return rewriter.notifyMatchFailure(
          coreBrick, "the q6_K GEMM core brick's block_index / strip_row_offset "
                     "must be region args 0 / 1");
    if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
        coreBrick.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          coreBrick, "the q6_K GEMM core brick's weight/activation bases must be "
                     "the loop-body's own repacked-weight / q8_Kx4-activation ABI "
                     "buffers");
    // q6_K (6-bit ql|qh) and q3_K (3-bit qs|hmask subtractive) SHARE the no-min GEMM
    // fold; the decode_model WHAT selects the per-family leaf. Both ship S6 STRIP-OUTER
    // TILED (the register-cliff lever DOES transfer -- board-remeasured [G8 stage-3]:
    // the strip-outer body drops q4_K's S6 accumulator-scoping onto the no-min fold,
    // clearing the ~2068-spill that pinned the old strip-inner emit to 0.15x). The
    // [ROLL] schedule axis is orthogonal and per-format measured: q6_K prefers UNROLLED
    // (small 8-position sub-block already fits I$; cold nr16 0.96/nr64 1.14), q3_K prefers
    // ROLLED (its 16-position unroll = 20933-line body thrashes I$ + residual spill;
    // rolled+S6 cold ~1.40 vs unrolled+S6 ~0.79) -- both S6-tiled, both byte-exact.
    if (coreBrick.getDecodeModel() != "q6_K" &&
        coreBrick.getDecodeModel() != "q3_K")
      return rewriter.notifyMatchFailure(
          coreBrick, "K-quant no-min repack GEMM decode_model not recognized "
                     "(expected \"q6_K\" or \"q3_K\")");
    std::optional<uint64_t> scalesOff = loopBody.getWeightScalesByteOffset();
    std::optional<uint64_t> qhOff = loopBody.getWeightQhByteOffset();
    std::optional<uint64_t> nSub = loopBody.getNSubblocks();
    if (!scalesOff || !qhOff || !nSub)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant no-min repack GEMM loop body requires the super-block "
                    "decode attrs weight_scales_byte_offset / weight_qh_byte_offset / "
                    "n_subblocks");
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    mlir::Value rowCount = valueMap.lookup(loopBody.getRowCount());
    mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(loopBody.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant no-min repack GEMM loop ABI operand unmapped");
    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!loopBody.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          loopBody, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
    // [GAP-EMIT-VSETVL-TAX] / [K-10] structural GEMM plan: the whole-K-nest [ROLL]
    // schedule axis (the *how*, never the *what*), resolved ONCE for the WHOLE no-min
    // K-quant GEMM family (q6_K AND q3_K share it -- the discriminant is the code-volume
    // vs I-cache budget FACT, NEVER the format spelling; the old "rolled is q6_K-only"
    // format-name hardcode is REMOVED). PREFER the explicit front-door main_term_form
    // stamp ("unrolled"|"rolled"), else the MEASURED-GATE default -- UNROLLED unless
    // code-volume>budget AND a board measurement records rolled beneficial (measured table
    // EMPTY today => UNROLLED; the q3_K/q6_K [ROLL] capability is WIRED SYMMETRICALLY by
    // the SAME code-volume predicate but activation is measured-gated, Stage-3 board A/B).
    // The GEMM tile is 4-column (activationInterleave=4), so the rolled register calculus
    // differs from the M=1 GEVM (a distinct structural plan, not a GEVM knob flip).
    // BYTE-EXACT across both schedules by construction (identical vwmacc16 accumulation
    // order).
    mlir::FailureOr<bool> rolled = requireMainTermRolled();
    if (mlir::failed(rolled))
      return rewriter.notifyMatchFailure(
          loopBody, "K-quant repack GEMM requires final main_term_form");
    bool rolledMainTerm = *rolled;
    // q3_K (3-bit subtractive qs|hmask): the qh slot carries the hmask plane offset.
    // RE-EMITs the byte-exact q3_K GEMM body (the no-min sibling of q6_K) -- now on the
    // SAME [ROLL] resolver as q6_K (rolledMainTerm passed in).
    if (coreBrick.getDecodeModel() == "q3_K")
      return emitRepackKQuantGemmBodyQ3K(
          rewriter, loc, weightBase, activationBase, output, rowCount, columnCount,
          outputRowStride, avlArg, sizeType, opName, role, coreLmul,
          static_cast<int64_t>(loopBody.getQk()),
          static_cast<int64_t>(loopBody.getWeightBlockStride()),
          static_cast<int64_t>(loopBody.getActivationBlockStride()),
          static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
          static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
          static_cast<int64_t>(*scalesOff), static_cast<int64_t>(*qhOff),
          static_cast<int64_t>(*nSub),
          static_cast<int64_t>(loopBody.getWeightInterleave()),
          static_cast<int64_t>(loopBody.getActivationInterleave()),
          static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm,
          selectedColGroupOuter);
    return emitRepackKQuantGemmBodyQ6K(
        rewriter, loc, weightBase, activationBase, output, rowCount, columnCount,
        outputRowStride, avlArg, sizeType, opName, role, coreLmul,
        static_cast<int64_t>(loopBody.getQk()),
        static_cast<int64_t>(loopBody.getWeightBlockStride()),
        static_cast<int64_t>(loopBody.getActivationBlockStride()),
        static_cast<int64_t>(loopBody.getWeightQuantByteOffset()),
        static_cast<int64_t>(loopBody.getActivationQuantByteOffset()),
        static_cast<int64_t>(*scalesOff), static_cast<int64_t>(*qhOff),
        static_cast<int64_t>(*nSub),
        static_cast<int64_t>(loopBody.getWeightInterleave()),
        static_cast<int64_t>(loopBody.getActivationInterleave()),
        static_cast<int64_t>(loopBody.getHalfLanes()), rolledMainTerm,
        selectedColGroupOuter);
  }

  // ---- Shape facts (the *how* -- LMUL / strip width / spill-avoiding
  // columnsPerPass -- never the *what*): the per-block strides drive the base
  // advance (loop shape); the within-block quant byte offsets driving the integer
  // core come from the CORE BRICK (the anti-bypass surface). numHalves ==
  // weight_interleave / half_lanes is the RUNTIME disjoint-strip count. ----
  int64_t qk = loopBody.getQk();
  int64_t weightInterleave = loopBody.getWeightInterleave();      // 16
  int64_t activationInterleave = loopBody.getActivationInterleave(); // 4
  int64_t half = loopBody.getHalfLanes();                         // 8 @128, 16 @256
  int64_t numHalves = weightInterleave / half;                    // 2 @128, 1 @256
  int64_t weightStride = loopBody.getWeightBlockStride();
  int64_t activationStride = loopBody.getActivationBlockStride();
  int64_t nibbleBytes = qk / 2;                                   // 16
  // The activation high-half int8 quants start after the low-half quants the
  // interleaved columns consume per nibble step (activationInterleave*nibbleBytes).
  int64_t activationHighRow = activationInterleave * nibbleBytes; // 64

  // The integer-core LMUL anchor (the *how*, never the *what*): "mf2" (the RVV1.0
  // fractional chain i8mf2 -> i16m1 -> i32m2 -> f32m2, f16 scale m1) or
  // "m1" (the RVV0.7 whole-LMUL chain i8m1 -> i16m2 -> i32m4 -> f32m4, f16 scale
  // m2). Only the type/callee LMUL suffixes change; numHalves, vl, every loop bound
  // and byte offset stay identical -- exactly the monolithic rung derivation. The
  // fold granularity columnsPerPass is 4 (all columns, one pass) for the mf2
  // fractional chain and 1 (one column per pass) for the m1 whole-LMUL chain (the
  // spill-avoiding form; see emitRepackGemmQ4_0Q8_0's columnsPerPass rationale).
  // Fail-closed final capability-fact read: the
  // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
  if (!loopBody.getIntegerCoreLmul())
    return rewriter.notifyMatchFailure(
        loopBody, "repack integer core requires an explicit integer_core_lmul "
              "capability fact (front door stamps it; no silent mf2 default)");
  llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
  llvm::StringRef l8 = coreLmul;                         // mf2 -> mf2; m1 -> m1
  llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";  // mf2 -> m1;  m1 -> m2
  llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";  // mf2 -> m2;  m1 -> m4
  int64_t columnsPerPass =
      (coreLmul == "m1") ? 1 : activationInterleave;      // 1 @rvv07; 4 @rvv1.0

  // ---- Region walk + region-driven gates (fail-closed, I7). The full-body region
  // carries the (block_index, strip_row_offset, columnsPerPass per-column vector
  // acc) entry args, ONE integer-core brick weft_rvv.repack_gemm_lane_wise_q4_x_i8_
  // dot producing columnsPerPass per-column sumi, columnsPerPass per-column dual-
  // fp16 scale FOLD bricks weft_rvv.repack_gemm_dual_fp16_scale_fold (each folding
  // one sumi + one carried acc), and a yield naming the columnsPerPass carried-out
  // vectors. Every brick is block_index + strip_row_offset tied (anti-bypass) and
  // dataflow-tied (the column-c fold consumes the integer brick's column-c sumi +
  // the column-c loop-carried acc; the yield names the folds' acc_next). ----
  mlir::Block &coreBlock = loopBody.getBody().front();
  weftrvv::TypedRepackGemmLoopYieldOp yieldOp;
  weftrvv::RepackGemmLaneWiseQ4Q8DotOp coreBrick;
  llvm::SmallVector<weftrvv::RepackGemmDualFp16ScaleFoldOp> foldBricks;
  loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
    if (auto o = llvm::dyn_cast<weftrvv::TypedRepackGemmLoopYieldOp>(bodyOp))
      yieldOp = o;
    else if (auto o = llvm::dyn_cast<weftrvv::RepackGemmLaneWiseQ4Q8DotOp>(bodyOp))
      coreBrick = o;
    else if (auto o =
                 llvm::dyn_cast<weftrvv::RepackGemmDualFp16ScaleFoldOp>(bodyOp))
      foldBricks.push_back(o);
  });
  if (!yieldOp)
    return rewriter.notifyMatchFailure(
        loopBody, "typed repack GEMM loop body requires the loop yield");
  if (static_cast<int64_t>(coreBlock.getNumArguments()) != columnsPerPass + 2)
    return rewriter.notifyMatchFailure(
        loopBody, "typed repack GEMM loop body region must carry the "
                  "(block_index, strip_row_offset, columnsPerPass per-column "
                  "vector acc) entry args");
  mlir::Value blockIndexArg = coreBlock.getArgument(0);
  mlir::Value stripOffsetArg = coreBlock.getArgument(1);

  // The integer CORE brick + its block_index + strip anti-bypass gate: the brick
  // must name the loop induction variable (region arg 0) as its block_index, the
  // region's runtime strip offset (region arg 1) as its strip_row_offset, and the
  // loop-body's own weight/activation ABI buffers as its bases (so the emit
  // provably addresses `base + block_index*stride (+ byte_offset)` at the runtime
  // strip, never the loop-invariant block 0 / hardcoded strip 0 / a foreign
  // buffer), and produce ONE per-column sumi per pass column (columnsPerPass
  // results).
  if (!coreBrick)
    return rewriter.notifyMatchFailure(
        loopBody, "repack GEMM loop body requires the region integer CORE brick "
                  "weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot (the one-strip "
                  "N-column per-block lane-wise nibble dot)");
  if (coreBrick.getBlockIndex() != blockIndexArg)
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack GEMM lane-wise dot brick's block_index must be the "
                   "loop induction variable (region arg 0)");
  if (coreBrick.getStripRowOffset() != stripOffsetArg)
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack GEMM lane-wise dot brick's strip_row_offset must be "
                   "the region's runtime strip offset (region arg 1)");
  if (coreBrick.getWeightBase() != loopBody.getWeightBase() ||
      coreBrick.getActivationBase() != loopBody.getActivationBase())
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack GEMM lane-wise dot brick's weight/activation bases "
                   "must be the loop-body's own repacked-weight / q8_0x4-activation "
                   "ABI buffers");
  if (static_cast<int64_t>(coreBrick.getNumResults()) != columnsPerPass)
    return rewriter.notifyMatchFailure(
        coreBrick, "the repack GEMM lane-wise dot brick must produce one per-column "
                   "sumi per pass column (columnsPerPass results)");

  // The columnsPerPass per-column dual-fp16 scale FOLD bricks, INDEXED by the
  // carried accumulator they consume: the column-c fold consumes region acc arg
  // (2 + c) AND the integer brick's column-c sumi (result c), is addressed off the
  // SAME block_index + strip + ABI buffers (anti-bypass), and produces the yield's
  // column-c acc_next. Every column shares the ONE within-block scale byte offset.
  if (static_cast<int64_t>(foldBricks.size()) != columnsPerPass)
    return rewriter.notifyMatchFailure(
        loopBody, "repack GEMM loop body requires one dual-fp16 scale FOLD brick "
                  "weft_rvv.repack_gemm_dual_fp16_scale_fold per pass column "
                  "(columnsPerPass bricks)");
  if (static_cast<int64_t>(yieldOp.getAccNext().size()) != columnsPerPass)
    return rewriter.notifyMatchFailure(
        yieldOp, "repack GEMM loop yield must name one carried-out per-column "
                 "accumulator per pass column (columnsPerPass acc_next)");
  llvm::SmallVector<weftrvv::RepackGemmDualFp16ScaleFoldOp> foldByColumn(
      columnsPerPass);
  for (int64_t c = 0; c < columnsPerPass; ++c) {
    mlir::Value accArg = coreBlock.getArgument(2 + c);
    weftrvv::RepackGemmDualFp16ScaleFoldOp fb;
    for (weftrvv::RepackGemmDualFp16ScaleFoldOp cand : foldBricks) {
      if (cand.getAcc() == accArg) {
        fb = cand;
        break;
      }
    }
    if (!fb)
      return rewriter.notifyMatchFailure(
          loopBody, "each per-column accumulator region argument must be consumed "
                    "by exactly one dual-fp16 scale FOLD brick");
    if (fb.getBlockIndex() != blockIndexArg)
      return rewriter.notifyMatchFailure(
          fb, "the repack GEMM scale-fold brick's block_index must be the loop "
              "induction variable (region arg 0)");
    if (fb.getStripRowOffset() != stripOffsetArg)
      return rewriter.notifyMatchFailure(
          fb, "the repack GEMM scale-fold brick's strip_row_offset must be the "
              "region's runtime strip offset (region arg 1)");
    if (fb.getWeightBase() != loopBody.getWeightBase() ||
        fb.getActivationBase() != loopBody.getActivationBase())
      return rewriter.notifyMatchFailure(
          fb, "the repack GEMM scale-fold brick's weight/activation bases must be "
              "the loop-body's own repacked-weight / q8_0x4-activation ABI "
              "buffers");
    if (fb.getSumi() != coreBrick.getResults()[c])
      return rewriter.notifyMatchFailure(
          fb, "the repack GEMM scale-fold brick for column c must consume the "
              "integer CORE brick's column-c per-column sumi (result c)");
    if (yieldOp.getAccNext()[c] != fb.getAccNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "the repack GEMM loop yield's column-c acc_next must be the "
                   "column-c scale-fold brick's folded-out accumulator");
    if (fb.getWeightScaleByteOffset() !=
            foldBricks.front().getWeightScaleByteOffset() ||
        fb.getActivationScaleByteOffset() !=
            foldBricks.front().getActivationScaleByteOffset())
      return rewriter.notifyMatchFailure(
          fb, "all per-column scale-fold bricks must share the ONE within-block "
              "fp16 scale byte offset");
    foldByColumn[c] = fb;
  }

  // ---- ABI operands (the same seven the monolithic repack GEMM reads; element
  // count n is the enclosing setvl AVL, exactly the GEVM emitter's derivation). ----
  mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
  mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
  mlir::Value output = valueMap.lookup(loopBody.getOutput());
  mlir::Value rowCount = valueMap.lookup(loopBody.getRowCount());
  mlir::Value columnCount = valueMap.lookup(loopBody.getColumnCount());
  mlir::Value outputRowStride = valueMap.lookup(loopBody.getOutputRowStride());
  if (!weightBase || !activationBase || !output || !rowCount || !columnCount ||
      !outputRowStride)
    return rewriter.notifyMatchFailure(loopBody,
                                       "repack GEMM loop ABI operand unmapped");

  llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();

  // The within-block quant byte offsets driving the integer core are SOURCED from
  // the CORE brick (a rewired offset changes the emitted addresses -- the
  // anti-bypass surface). The fp16 scale byte offsets are the fold bricks' declared
  // facts (the shared fold leaf hardcodes the block-leading scale d, byte-exact to
  // the monolith); they are gate-verified consistent above.
  int64_t weightQuantOffset = coreBrick.getWeightQuantByteOffset();
  int64_t activationQuantOffset = coreBrick.getActivationQuantByteOffset();
  // q4_1 decode-leaf facts from the region BRICKS (anti-bypass surface): the
  // UNSIGNED-nibble flag from the CORE brick + the single MIN-fold offset pair from
  // the per-column FOLD bricks (-1 sentinel = the q4_0 no-min fold, byte-identical).
  bool unsignedNibble = coreBrick.getWeightNibbleUnsigned();
  // q8_0 FULL-int8 decode selector sourced from the GEMM CORE brick.
  bool coreFullI8 = coreBrick.getWeightFullI8();
  // q5_0 5th-bit (qh) decode facts sourced from the GEMM CORE brick.
  bool coreHasQh = coreBrick.getWeightQhByteOffset().has_value();
  int64_t coreWeightQhOffset =
      coreHasQh ? static_cast<int64_t>(*coreBrick.getWeightQhByteOffset()) : 0;
  int64_t coreOffsetBias =
      coreBrick.getWeightOffsetBias().has_value()
          ? static_cast<int64_t>(*coreBrick.getWeightOffsetBias())
          : 0;
  int64_t weightMinOffset =
      foldByColumn[0].getWeightMinByteOffset().has_value()
          ? static_cast<int64_t>(*foldByColumn[0].getWeightMinByteOffset())
          : -1;
  int64_t activationSumOffset =
      foldByColumn[0].getActivationSumByteOffset().has_value()
          ? static_cast<int64_t>(*foldByColumn[0].getActivationSumByteOffset())
          : -1;

  mlir::Type f32AccType =
      emitc::OpaqueType::get(ctx, ("vfloat32" + l32 + "_t").str());
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type weightPtrType = weightBase.getType();
  mlir::Type activationPtrType = activationBase.getType();
  mlir::Type floatPtrType = output.getType();

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto step = [&](llvm::StringRef s) {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // The active vl is the compile-time-constant strip width (half_lanes e16m1 lanes:
  // 16 for the one-strip VLEN=256/RVV0.7 form, 8 for the two-halves VLEN=128 form),
  // exactly as the monolithic repack GEMM runs every intrinsic.
  mlir::Value vl8 = sizeLit(half);

  // size_t nb = n / QK; nr_groups = nr / 4; nc_groups = nc / 16.
  step("block_count");
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
  step("row_group_count");
  mlir::Value nrGroups = rewriter.create<emitc::DivOp>(
      loc, sizeType, rowCount, sizeLit(activationInterleave));
  step("col_group_count");
  mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
      loc, sizeType, columnCount, sizeLit(weightInterleave));

  // ===== [loop-order REALIZE] SEL-1 loop-order schedule axis (PURE REALIZE) =====
  // The activation ROW-GROUP loop (nr/4) and the weight COLUMN-GROUP loop (nc/16)
  // are INDEPENDENT -- every out[y,x] is a private K-accumulation -- so either
  // nesting order yields BYTE-IDENTICAL results and an identical hot inner core.
  // Which loop is OUTER is the selected schedule axis resolved above
  // from the front-door loop_order stamp (row-group-OUTER == the
  // M1-committed flat default; col-group-OUTER holds the DRAM-dominant repacked
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
  // arm reproduces the M1-committed single fixed flat nest op-for-op.
  auto emitTile = [&](mlir::Value x, mlir::Value y, mlir::Value bGroup,
                      mlir::Value aGroup) {

      // ===== Strip loop: for (h = 0; h < num_halves; ++h) {roff = h*half} =====
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

        // ===== Activation-column-PASS loop (compile-time, C++): the columns
        // [cLo, cLo+columnsPerPass) folded in this pass over the block loop.
        for (int64_t cLo = 0; cLo < activationInterleave;
             cLo += columnsPerPass) {
          int64_t cHi = cLo + columnsPerPass;
          // vfloat32{m2,m4}_t sumf_{cLo..cHi} = vfmv_v_f(0.0f, vl);
          std::string fmvCallee = riscvIntrinsicName("vfmv_v_f", 32, l32, "f32");
          llvm::SmallVector<mlir::Value> sumf(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            sumf[c] = emitOpaqueCallBuilt(
                rewriter, loc, f32AccType, fmvCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  mlir::Value zero =
                      rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                          .getResult();
                  return {zero, vl8};
                });
          }
          // Mutable f32 accumulator lvalues (the inner block loop carries them;
          // the region's per-column acc entry args (2+c) lower to these lvalues).
          llvm::SmallVector<mlir::Value> sumfVar(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            auto v = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32AccType),
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

            // const uint8_t *bl = b + l*288;   const uint8_t *al = a + l*136;
            // The per-block bases advance off the loop induction variable `l` (==
            // the bricks' gated block_index) by the loop-op strides -- the
            // block_index-tied anti-bypass addressing, byte-exact to the monolith.
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

            // ===== The region integer CORE: the weft_rvv.repack_gemm_lane_wise_
            // q4_x_i8_dot brick lowers to the SHARED emitRepackGemmQ4LaneWise
            // IntegerCore leaf (seed per-column i16 lo/hi -> nibble-step vwmacc
            // loop over ONE strip at the runtime roff and the [cLo,cHi) interleaved
            // columns -> per-column lo/hi vwadd combine), byte-exact to the
            // monolith's integer part, producing the columnsPerPass per-column
            // sumi. The within-block quant byte offsets are SOURCED from the CORE
            // brick (the anti-bypass surface). =====
            RepackGemmQ4IntegerCoreContext coreCx{
                opName, role, l8, l16, l32, nibbleBytes, weightInterleave,
                weightQuantOffset, activationQuantOffset, activationInterleave,
                activationHighRow, vl8, sizeType};
            coreCx.unsignedNibble = unsignedNibble;
            coreCx.hasQh = coreHasQh;
            coreCx.weightQhByteOffset = coreWeightQhOffset;
            coreCx.offsetBias = coreOffsetBias;
            coreCx.fullI8 = coreFullI8;
            llvm::SmallVector<mlir::Value> sumi32 =
                emitRepackGemmQ4LaneWiseIntegerCore(rewriter, loc, coreCx, bl, al,
                                                    roff, cLo, cHi);

            // ===== The region dual-fp16 scale FOLD: the columnsPerPass
            // weft_rvv.repack_gemm_dual_fp16_scale_fold bricks lower through ONE
            // call to the SHARED emitRepackGemmDualFp16ScaleFold leaf (vle16 the
            // per-strip weight scales -> per column widen d = vfwmul(b_d, act_d),
            // convert the column sumi, vfmacc into sumf_c; load sumf_c, fold, assign
            // back), byte-exact to the monolith's fold part. =====
            RepackGemmDualFp16ScaleFoldContext foldCx{opName, role, l16, l32,
                                                      vl8, sizeType};
            foldCx.weightMinByteOffset = weightMinOffset;
            foldCx.activationSumByteOffset = activationSumOffset;
            emitRepackGemmDualFp16ScaleFold(rewriter, loc, foldCx, bl, al, roff,
                                            sumi32, sumfVar, cLo, cHi);
          }

          // vse32(s + (y*4 + c)*bs + x*16 + roff, sumf_c, vl);  -- the 4x8 store.
          std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
          for (int64_t c = cLo; c < cHi; ++c) {
            step("output_addr");
            // row = y*4 + c;  off = row*bs + x*16 + roff.
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
                rewriter.create<emitc::LoadOp>(loc, f32AccType, sumfVar[c])
                    .getResult();
            emitOpaqueCallVoid(rewriter, loc, vseCallee,
                               mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
          }
        } // end activation-column-PASS loop (cLo)
      }
  };  // end emitTile (byte-identical body for both loop orders)

  if (selectedColGroupOuter) {
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

  return mlir::success();
}

// q5_0 16x1-REPACKED single-column GEMV (decode). q5_0 = q4_0 + the 5th high
// bit. The weight side is block_q5_0x16 (16 interleaved rows across 16 lanes,
// dot accumulates LANE-WISE via vwmacc, NO per-block vredsum): RAW nibbles at
// +32 (SAME 16-way interleave as q4_0x16, but no ^0x88 bake -- the bias lives
// in the assembled 5-bit field), and a 64-byte TRANSPOSED bit-packed qh region
// at +288 carrying one 16-bit mask per element step (mask[e] bit b = block b's
// NON-inverted qh bit for element e). Per nibble step i the lane decode expands
// the two masks (low element i, high element i+16), assembles the UNSIGNED
// A = nibble | (qh_bit<<4) in [0,31], reinterprets u8->i8, then subtracts 16
// (vsub) -- the q5_0 offset-binary -16: ((nibble | (qh_bit<<4)) - 16), the
// PROVEN block-dot reconstruct. Activation (plain q8_0 stride 34) + dual-fp16
// scale fold are byte-identical to emitRepackGemvQ4_0Q8_0.
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ5_0Q8_0(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlRepackGemvQ50Q80Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<weftrvv::GgmlRepackGemvQ50Q80Op>(op))
        gemv = g;
    }
    if (!gemv)
      return rewriter.notifyMatchFailure(scope, "repack-gemv body missing op");

    mlir::Value weightBase = valueMap.lookup(gemv.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemv.getActivationBase());
    mlir::Value output = valueMap.lookup(gemv.getOutput());
    mlir::Value columnCount = valueMap.lookup(gemv.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(gemv, "repack-gemv ABI operand unmapped");

    llvm::StringRef opName = gemv.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!gemv.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          gemv, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *gemv.getIntegerCoreLmul();
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
    // The 5-bit assembly runs in UNSIGNED u8 ([0,31]) then reinterprets to i8 and
    // subtracts 16 -- the PROVEN block-dot decode path (vsrl/vand/vor are unsigned
    // only). The qh lane-expansion works in the l16 width (so a full 16-bit mask
    // fits at half_lanes==16) and narrows the {0,16} term to u8.
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    int64_t qk = gemv.getQk();
    int64_t weightStride = gemv.getWeightBlockStride();
    int64_t activationStride = gemv.getActivationBlockStride();
    int64_t weightQuantOffset = gemv.getWeightQuantByteOffset();
    int64_t weightQhOffset = gemv.getWeightQhByteOffset();   // 288
    int64_t activationQuantOffset = gemv.getActivationQuantByteOffset();
    int64_t weightInterleave = gemv.getWeightInterleave();   // 16
    int64_t half = gemv.getHalfLanes();                      // 8 @128, 16 @256
    int64_t numHalves = weightInterleave / half;             // 2 @128, 1 @256
    int64_t nibbleBytes = qk / 2;                            // 16 nibble bytes
    int64_t activationHighRow = nibbleBytes;                 // 16

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemv, "repack-gemv output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    mlir::Value vl8 = sizeLit(half);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // A typed u8 contiguous nibble sub-load: __riscv_vle8_v_u8<l8>(...). The
    // whole nibble/qh assembly runs UNSIGNED (the block-dot path).
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // The RAW nibble extract (no ^0x88 bake): the q5_0 decode follows the PROVEN
    // block-dot path (RVVToEmitC.cpp fifthBitLane / reinterpretBias): isolate the
    // UNSIGNED [0,15] nibble, OR in (qhbit<<4) -> [0,31], reinterpret u8->i8, then
    // apply the `-16` bias via signed vsub. lo_nib = (b & 0x0F); hi_nib = (b >>
    // 4) [both unsigned].
    std::string andCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string srlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    std::string orCallee = ("__riscv_vor_vv_u8" + l8).str();
    std::string reinterpCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    std::string subCallee = ("__riscv_vsub_vx_i8" + l8).str();
    mlir::Value four = sizeLit(4);
    mlir::Value mask0F = sizeLit(15);
    mlir::Value bias16 = sizeLit(16);
    auto vandVx = [&](mlir::Value v, mlir::Value imm) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, u8mf2Type, andCallee,
                            mlir::ValueRange{v, imm, vl8}, opName, role);
    };
    // lo nibble [0,15]: vand(b, 0x0F).
    auto nibbleLo = [&](mlir::Value packed) -> mlir::Value {
      return vandVx(packed, mask0F);
    };
    // hi nibble [0,15]: vsrl(b, 4) (logical shift -> high nibble in low 4 bits).
    auto nibbleHi = [&](mlir::Value packed) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, u8mf2Type, srlCallee,
                            mlir::ValueRange{packed, four, vl8}, opName, role);
    };

    // The transposed bit-packed qh mask scalar read: a 16-bit mask for one
    // element step (low element i at qh+i*2, high element i+16 at qh+(16+i)*2).
    // Read as (uint16_t)*(const uint16_t *). The per-strip bit selection is done
    // in the VECTOR expansion (vid + h*half), NOT a scalar pre-shift -- so the
    // full 16-bit mask flows in unchanged.
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    auto qhMaskScalar = [&](mlir::Value bl, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, bl, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, u16ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("qh_mask_scalar"));
    };
    // Expand a strip's qh mask into a per-lane 5th bit term {0,16} in u8: splat
    // the 16-bit mask into u16 lanes, vsrl by (vid + h*half) so lane l of strip h
    // reads bit (l + h*half), vand 1 -> bit per lane, vsll 4 -> {0,16}, narrow
    // u16->u8 -- the (qhbit<<4) term the assemble step ORs into the nibble (the
    // SAME shape as the proven block-dot fifthBitLane).
    std::string vidCallee = ("__riscv_vid_v_u16" + l16).str();
    std::string vmvU16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "u16");
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    std::string vsrlVvCallee = ("__riscv_vsrl_vv_u16" + l16).str();
    std::string vandU16Callee = ("__riscv_vand_vx_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vncvtCallee = ("__riscv_vncvt_x_x_w_u8" + l8).str();
    auto expandQhBit = [&](mlir::Value maskScalar,
                           int64_t laneShift) -> mlir::Value {
      mlir::Value splat =
          emitOpaqueCall(rewriter, loc, u16m1Type, vmvU16Callee,
                         mlir::ValueRange{maskScalar, vl8}, opName, role);
      mlir::Value vid =
          emitOpaqueCall(rewriter, loc, u16m1Type, vidCallee,
                         mlir::ValueRange{vl8}, opName, role);
      if (laneShift != 0) {
        vid = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {vid, sizeLit(laneShift), vl8};
            });
      }
      mlir::Value shifted =
          emitOpaqueCall(rewriter, loc, u16m1Type, vsrlVvCallee,
                         mlir::ValueRange{splat, vid, vl8}, opName, role);
      mlir::Value bit = emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vandU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {shifted, sizeLit(1), vl8};
          });
      // bit -> {0,16}: vsll 4 (in u16), then narrow u16->i8 (value 0/16
      // unchanged, both fit i8).
      mlir::Value bit16 =
          emitOpaqueCall(rewriter, loc, u16m1Type, vsllU16Callee,
                         mlir::ValueRange{bit, four, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, u8mf2Type, vncvtCallee,
                            mlir::ValueRange{bit16, vl8}, opName, role);
    };
    // Assemble the 5-bit weight: A = nibble | (qhbit<<4) -> u8 [0,31], reinterpret
    // u8->i8 (value-identity for 0..31), then apply the offset-binary `-16` bias
    // via signed vsub -> i8 [-16,15]. This is the PROVEN block-dot reconstruct
    // (fifthBitLane + reinterpretBias).
    auto assemble5 = [&](mlir::Value nibble, mlir::Value bit16) -> mlir::Value {
      mlir::Value a =
          emitOpaqueCall(rewriter, loc, u8mf2Type, orCallee,
                         mlir::ValueRange{nibble, bit16, vl8}, opName, role);
      mlir::Value as =
          emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpCallee,
                         mlir::ValueRange{a}, opName, role);
      return emitOpaqueCall(rewriter, loc, i8mf2Type, subCallee,
                            mlir::ValueRange{as, bias16, vl8}, opName, role);
    };
    // A scalar i8 read of the plain q8_0 activation quant byte a_ptr[l].qs[k].
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
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmaccCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
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
        llvm::SmallVector<mlir::Value> sumiLoVar, sumiHiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto vlo = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i16m1Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, vlo, seedI16());
          sumiLoVar.push_back(vlo);
          auto vhi = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i16m1Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, vhi, seedI16());
          sumiHiVar.push_back(vhi);
        }

        // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
        auto nibLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(nibbleBytes), sizeLit(1),
            /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard ng(rewriter);
          rewriter.setInsertionPointToStart(nibLoop.getBody());
          mlir::Value i = nibLoop.getInductionVar();

          // num_halves disjoint nibble sub-loads, strip h at qs[i*16+h*half].
          step("weight_nibble_addr");
          mlir::Value i16 = rewriter.create<emitc::MulOp>(
              loc, sizeType, i, sizeLit(weightInterleave));
          mlir::Value qsOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQuantOffset), i16);
          llvm::SmallVector<mlir::Value> wByteOff;
          for (int64_t h = 0; h < numHalves; ++h) {
            if (h == 0)
              wByteOff.push_back(qsOff);
            else
              wByteOff.push_back(rewriter.create<emitc::AddOp>(
                  loc, sizeType, qsOff, sizeLit(h * half)));
          }
          llvm::SmallVector<mlir::Value> packed;
          for (int64_t h = 0; h < numHalves; ++h)
            packed.push_back(loadNibbles(bl, wByteOff[h]));

          // The transposed qh masks: low element i at qh + i*2, high element
          // i+16 at qh + (16+i)*2. Byte offsets are i*2 + qhBase (lo) and
          // (i+16)*2 + qhBase (hi). Per strip h the mask is pre-shifted h*half.
          step("qh_lo_addr");
          mlir::Value iTwo = rewriter.create<emitc::MulOp>(
              loc, sizeType, i, sizeLit(2));
          mlir::Value qhLoOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQhOffset), iTwo);
          step("qh_hi_addr");
          mlir::Value qhHiBase = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQhOffset),
              sizeLit(activationHighRow * 2));
          mlir::Value qhHiOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, qhHiBase, iTwo);

          // DECODE phase: per strip, assemble lo then hi 5-bit weights. The qh
          // mask is read ONCE per element step (lo, hi); each strip selects its
          // 8/16 lanes via the (vid + h*half) shift in expandQhBit.
          mlir::Value loMaskS = qhMaskScalar(bl, qhLoOff);
          mlir::Value hiMaskS = qhMaskScalar(bl, qhHiOff);
          llvm::SmallVector<mlir::Value> bLo, bHi;
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value loBit = expandQhBit(loMaskS, h * half);
            bLo.push_back(assemble5(nibbleLo(packed[h]), loBit));
            mlir::Value hiBit = expandQhBit(hiMaskS, h * half);
            bHi.push_back(assemble5(nibbleHi(packed[h]), hiBit));
          }

          // Single activation column (SHARED across strips): al.qs[i] (low),
          // al.qs[16+i] (high).
          step("act_quant_addr_lo");
          mlir::Value loOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset), i);
          mlir::Value aLo = i8Read(al, loOff);
          step("act_quant_addr_hi");
          mlir::Value hiBase = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset),
              sizeLit(activationHighRow));
          mlir::Value hiOff =
              rewriter.create<emitc::AddOp>(loc, sizeType, hiBase, i);
          mlir::Value aHi = i8Read(al, hiOff);

          // ACCUMULATE phase: per strip, lane-wise vwmacc lo then hi.
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value curLo =
                rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiLoVar[h],
                                             vwmacc(curLo, aLo, bLo[h]));
            mlir::Value curHi =
                rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiHiVar[h],
                                             vwmacc(curHi, aHi, bHi[h]));
          }
        }

        // const vint32m2_t sumi_h = vwadd_vv(sumi_h_lo, sumi_h_hi, vl).
        std::string vwaddCallee = ("__riscv_vwadd_vv_i32" + l32).str();
        auto combine = [&](mlir::Value loVar, mlir::Value hiVar) -> mlir::Value {
          mlir::Value lo =
              rewriter.create<emitc::LoadOp>(loc, i16m1Type, loVar).getResult();
          mlir::Value hi =
              rewriter.create<emitc::LoadOp>(loc, i16m1Type, hiVar).getResult();
          return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                                mlir::ValueRange{lo, hi, vl8}, opName, role);
        };
        llvm::SmallVector<mlir::Value> sumi;
        for (int64_t h = 0; h < numHalves; ++h)
          sumi.push_back(combine(sumiLoVar[h], sumiHiVar[h]));

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
        llvm::SmallVector<mlir::Value> bD;
        for (int64_t h = 0; h < numHalves; ++h)
          bD.push_back(loadScales(h * half));

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
        auto fold = [&](mlir::Value bD, mlir::Value sumi,
                        mlir::Value sumfVar) {
          mlir::Value dC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bD, aD, vl8}, opName, role);
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumi, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar)
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                             mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar, nextF);
        };
        for (int64_t h = 0; h < numHalves; ++h)
          fold(bD[h], sumi[h], sumfVar[h]);
      }

      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      auto storeHalf = [&](mlir::Value sumfVar, int64_t laneOff) {
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
            rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar).getResult();
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
      };
      for (int64_t h = 0; h < numHalves; ++h)
        storeHalf(sumfVar[h], h * half);
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
    mlir::Value zeroLane =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
    mlir::Value resultTok =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                         seedCallee,
                                         mlir::ValueRange{zeroLane, sizeLit(1)})
            .getResult(0);
    valueMap[gemv.getResult()] = resultTok;
    return mlir::success();
  }

// q5_1 GEVM emitter — the UNION of the q5_0 GEVM (five-bit transposed-qh decode)
// and the q4_1 GEVM (Family-B scale+MIN dual fold, plain q8_1 activation). q5_1 is
// q4_1 plus the 5th high bit: the integer core assembles the UNSIGNED 5-bit weight
// A = nibble | (qh_bit<<4) in [0,31] (SAME qh source as q5_0) but WITHOUT q5_0's
// `-16` offset-binary bias (q5_1 is asymmetric; the bias lives in the MIN scale),
// then feeds the SAME lane-wise vwmacc + the q4_1 dual fold. Byte-exact vs ggml
// q5_1 by construction.
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ5_1Q8_1(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlRepackGemvQ51Q81Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<weftrvv::GgmlRepackGemvQ51Q81Op>(op))
        gemv = g;
    }
    if (!gemv)
      return rewriter.notifyMatchFailure(scope, "repack-gemv-q5_1 body missing op");

    mlir::Value weightBase = valueMap.lookup(gemv.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemv.getActivationBase());
    mlir::Value output = valueMap.lookup(gemv.getOutput());
    mlir::Value columnCount = valueMap.lookup(gemv.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(gemv,
                                         "repack-gemv-q5_1 ABI operand unmapped");

    llvm::StringRef opName = gemv.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one
    // notch i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!gemv.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          gemv, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *gemv.getIntegerCoreLmul();
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
    // The 5-bit assembly runs in UNSIGNED u8 ([0,31]) then reinterprets to i8 --
    // WITHOUT the q5_0 `-16` bias (q5_1 is asymmetric). The qh lane-expansion
    // works in the l16 width and narrows the {0,16} term to u8.
    mlir::Type u8mf2Type =
        emitc::OpaqueType::get(ctx, ("vuint8" + l8 + "_t").str());
    mlir::Type u16m1Type =
        emitc::OpaqueType::get(ctx, ("vuint16" + l16 + "_t").str());
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked q5_1 block-format structural facts (I4 mirror, pinned by
    // the verifier): QK=32, block_q5_1x16 weight stride 384 (16 d + 16 m + 256
    // nibble bytes + 64 transposed qh bytes), block_q8_1 activation stride 36,
    // weight nibbles at +64, the per-row MIN strip at +32, the transposed qh masks
    // at +320, activation quants at +4, activation scaled-sum s at +2, 16 weight
    // rows per group, the VLEN-derived e16m1 half width.
    int64_t qk = gemv.getQk();
    int64_t weightStride = gemv.getWeightBlockStride();
    int64_t activationStride = gemv.getActivationBlockStride();
    int64_t weightQuantOffset = gemv.getWeightQuantByteOffset();   // 64
    int64_t weightQhOffset = gemv.getWeightQhByteOffset();         // 320
    int64_t weightMinOffset = gemv.getWeightMinByteOffset();       // 32
    int64_t activationQuantOffset = gemv.getActivationQuantByteOffset(); // 4
    int64_t activationSumOffset = gemv.getActivationSumByteOffset(); // 2
    int64_t weightInterleave = gemv.getWeightInterleave();   // 16
    int64_t half = gemv.getHalfLanes();                      // 8 @128, 16 @256
    int64_t numHalves = weightInterleave / half;             // 2 @128, 1 @256
    int64_t nibbleBytes = qk / 2;                            // 16 nibble bytes
    int64_t activationHighRow = nibbleBytes;                 // 16

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemv, "repack-gemv-q5_1 output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    mlir::Value vl8 = sizeLit(half);

    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // A typed u8 contiguous nibble sub-load (RAW nibbles, the block-dot path).
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, l8, "u8");
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, u8mf2Type, u8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // lo_nib = (b & 0x0F); hi_nib = (b >> 4) [both unsigned].
    std::string andCallee = ("__riscv_vand_vx_u8" + l8).str();
    std::string srlCallee = ("__riscv_vsrl_vx_u8" + l8).str();
    std::string orCallee = ("__riscv_vor_vv_u8" + l8).str();
    std::string reinterpCallee =
        ("__riscv_vreinterpret_v_u8" + l8 + "_i8" + l8).str();
    mlir::Value four = sizeLit(4);
    mlir::Value mask0F = sizeLit(15);
    auto vandVx = [&](mlir::Value v, mlir::Value imm) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, u8mf2Type, andCallee,
                            mlir::ValueRange{v, imm, vl8}, opName, role);
    };
    auto nibbleLo = [&](mlir::Value packed) -> mlir::Value {
      return vandVx(packed, mask0F);
    };
    auto nibbleHi = [&](mlir::Value packed) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, u8mf2Type, srlCallee,
                            mlir::ValueRange{packed, four, vl8}, opName, role);
    };

    // The transposed bit-packed qh mask scalar read: one 16-bit mask per element
    // step (low element i at qh+i*2, high element i+16 at qh+(16+i)*2). The
    // per-strip bit selection is done in the VECTOR expansion (vid + h*half).
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    auto qhMaskScalar = [&](mlir::Value bl, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, bl, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i32Type, u16ReadCallee,
                            mlir::ValueRange{cast}, opName, role,
                            llvm::StringRef("qh_mask_scalar"));
    };
    // Expand a strip's qh mask into a per-lane 5th bit term {0,16} in u8: splat
    // the 16-bit mask into u16 lanes, vsrl by (vid + h*half) so lane l of strip h
    // reads bit (l + h*half), vand 1 -> bit per lane, vsll 4 -> {0,16}, narrow
    // u16->u8. IDENTICAL to the q5_0 fifthBitLane.
    std::string vidCallee = ("__riscv_vid_v_u16" + l16).str();
    std::string vmvU16Callee = riscvIntrinsicName("vmv_v_x", 16, l16, "u16");
    std::string vaddU16Callee = ("__riscv_vadd_vx_u16" + l16).str();
    std::string vsrlVvCallee = ("__riscv_vsrl_vv_u16" + l16).str();
    std::string vandU16Callee = ("__riscv_vand_vx_u16" + l16).str();
    std::string vsllU16Callee = ("__riscv_vsll_vx_u16" + l16).str();
    std::string vncvtCallee = ("__riscv_vncvt_x_x_w_u8" + l8).str();
    auto expandQhBit = [&](mlir::Value maskScalar,
                           int64_t laneShift) -> mlir::Value {
      mlir::Value splat =
          emitOpaqueCall(rewriter, loc, u16m1Type, vmvU16Callee,
                         mlir::ValueRange{maskScalar, vl8}, opName, role);
      mlir::Value vid =
          emitOpaqueCall(rewriter, loc, u16m1Type, vidCallee,
                         mlir::ValueRange{vl8}, opName, role);
      if (laneShift != 0) {
        vid = emitOpaqueCallBuilt(
            rewriter, loc, u16m1Type, vaddU16Callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {vid, sizeLit(laneShift), vl8};
            });
      }
      mlir::Value shifted =
          emitOpaqueCall(rewriter, loc, u16m1Type, vsrlVvCallee,
                         mlir::ValueRange{splat, vid, vl8}, opName, role);
      mlir::Value bit = emitOpaqueCallBuilt(
          rewriter, loc, u16m1Type, vandU16Callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {shifted, sizeLit(1), vl8};
          });
      mlir::Value bit16 =
          emitOpaqueCall(rewriter, loc, u16m1Type, vsllU16Callee,
                         mlir::ValueRange{bit, four, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, u8mf2Type, vncvtCallee,
                            mlir::ValueRange{bit16, vl8}, opName, role);
    };
    // Assemble the UNSIGNED 5-bit weight: A = nibble | (qhbit<<4) -> u8 [0,31],
    // reinterpret u8->i8 (value-identity for 0..31). UNLIKE the q5_0 sibling there
    // is NO `-16` bias vsub (q5_1 is asymmetric; the bias lives in the MIN scale).
    auto assemble5Unsigned = [&](mlir::Value nibble,
                                 mlir::Value bit16) -> mlir::Value {
      mlir::Value a =
          emitOpaqueCall(rewriter, loc, u8mf2Type, orCallee,
                         mlir::ValueRange{nibble, bit16, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, i8mf2Type, reinterpCallee,
                            mlir::ValueRange{a}, opName, role);
    };
    // A scalar i8 read of the plain q8_1 activation quant byte a_ptr[l].qs[k].
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
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmaccCallee,
                            mlir::ValueRange{acc, scalar, vec, vl8}, opName,
                            role);
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

      // const uint8_t *b = vx + x*nb*384;  (the q5_1x16 column group base).
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

        // const uint8_t *bl = b + l*384;   const uint8_t *al = a + l*36;
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
        llvm::SmallVector<mlir::Value> sumiLoVar, sumiHiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto vlo = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i16m1Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, vlo, seedI16());
          sumiLoVar.push_back(vlo);
          auto vhi = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i16m1Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, vhi, seedI16());
          sumiHiVar.push_back(vhi);
        }

        // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
        auto nibLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(nibbleBytes), sizeLit(1),
            /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard ng(rewriter);
          rewriter.setInsertionPointToStart(nibLoop.getBody());
          mlir::Value i = nibLoop.getInductionVar();

          // num_halves disjoint nibble sub-loads, strip h at qs[i*16+h*half].
          step("weight_nibble_addr");
          mlir::Value i16 = rewriter.create<emitc::MulOp>(
              loc, sizeType, i, sizeLit(weightInterleave));
          mlir::Value qsOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQuantOffset), i16);
          llvm::SmallVector<mlir::Value> wByteOff;
          for (int64_t h = 0; h < numHalves; ++h) {
            if (h == 0)
              wByteOff.push_back(qsOff);
            else
              wByteOff.push_back(rewriter.create<emitc::AddOp>(
                  loc, sizeType, qsOff, sizeLit(h * half)));
          }
          llvm::SmallVector<mlir::Value> packed;
          for (int64_t h = 0; h < numHalves; ++h)
            packed.push_back(loadNibbles(bl, wByteOff[h]));

          // The transposed qh masks: low element i at qh + i*2, high element
          // i+16 at qh + (16+i)*2.
          step("qh_lo_addr");
          mlir::Value iTwo = rewriter.create<emitc::MulOp>(
              loc, sizeType, i, sizeLit(2));
          mlir::Value qhLoOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQhOffset), iTwo);
          step("qh_hi_addr");
          mlir::Value qhHiBase = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQhOffset),
              sizeLit(activationHighRow * 2));
          mlir::Value qhHiOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, qhHiBase, iTwo);

          // DECODE phase: per strip, assemble lo then hi 5-bit weights (UNSIGNED,
          // NO `-16` bias).
          mlir::Value loMaskS = qhMaskScalar(bl, qhLoOff);
          mlir::Value hiMaskS = qhMaskScalar(bl, qhHiOff);
          llvm::SmallVector<mlir::Value> bLo, bHi;
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value loBit = expandQhBit(loMaskS, h * half);
            bLo.push_back(assemble5Unsigned(nibbleLo(packed[h]), loBit));
            mlir::Value hiBit = expandQhBit(hiMaskS, h * half);
            bHi.push_back(assemble5Unsigned(nibbleHi(packed[h]), hiBit));
          }

          // Single activation column (SHARED across strips): al.qs[i] (low),
          // al.qs[16+i] (high), at activation_quant_byte_offset +4.
          step("act_quant_addr_lo");
          mlir::Value loOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset), i);
          mlir::Value aLo = i8Read(al, loOff);
          step("act_quant_addr_hi");
          mlir::Value hiBase = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset),
              sizeLit(activationHighRow));
          mlir::Value hiOff =
              rewriter.create<emitc::AddOp>(loc, sizeType, hiBase, i);
          mlir::Value aHi = i8Read(al, hiOff);

          // ACCUMULATE phase: per strip, lane-wise vwmacc lo then hi.
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value curLo =
                rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiLoVar[h],
                                             vwmacc(curLo, aLo, bLo[h]));
            mlir::Value curHi =
                rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiHiVar[h],
                                             vwmacc(curHi, aHi, bHi[h]));
          }
        }

        // const vint32m2_t sumi_h = vwadd_vv(sumi_h_lo, sumi_h_hi, half).
        std::string vwaddCallee = ("__riscv_vwadd_vv_i32" + l32).str();
        auto combine = [&](mlir::Value loVar, mlir::Value hiVar) -> mlir::Value {
          mlir::Value lo =
              rewriter.create<emitc::LoadOp>(loc, i16m1Type, loVar).getResult();
          mlir::Value hi =
              rewriter.create<emitc::LoadOp>(loc, i16m1Type, hiVar).getResult();
          return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                                mlir::ValueRange{lo, hi, vl8}, opName, role);
        };
        llvm::SmallVector<mlir::Value> sumi;
        for (int64_t h = 0; h < numHalves; ++h)
          sumi.push_back(combine(sumiLoVar[h], sumiHiVar[h]));

        // Per-row delta d strips (at byte 0 + h*half*2) and MIN m strips (at byte
        // weightMinOffset + h*half*2) -- the Family-B dual scale, exactly as q4_1.
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
        llvm::SmallVector<mlir::Value> bD, bM;
        for (int64_t h = 0; h < numHalves; ++h)
          bD.push_back(loadF16Strip(0, h * half));
        for (int64_t h = 0; h < numHalves; ++h)
          bM.push_back(loadF16Strip(weightMinOffset, h * half));

        // The single activation scale d_y = *(const _Float16 *)&al.d and the
        // scaled-sum s_y = *(const _Float16 *)&al.s (at +2).
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
        mlir::Value aS = emitOpaqueCallBuilt(
            rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aSFull = rewriter.create<emitc::AddOp>(
                  loc, activationPtrType, al, sizeLit(activationSumOffset));
              mlir::Value aSCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, aSFull)
                      .getResult();
              return {aSCast};
            },
            llvm::StringRef("act_sum_scalar"));

        // Per strip, ggml's q5_1 fold sumf += (d_x*d_y)*sumi + m_x*s_y:
        //   d_h  = vfwmul_vf(b_d_h, d_y, half);            // (d_x * d_y)
        //   sumf_h = vfmacc_vv(sumf_h, vfcvt(sumi_h), d_h);  // + (d_x*d_y)*sumi
        //   m_h  = vfwmul_vf(b_m_h, s_y, half);            // (m_x * s_y)
        //   sumf_h = vfadd_vv(sumf_h, m_h, half);          // + m_x*s_y
        std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        std::string vfaddCallee = ("__riscv_vfadd_vv_f32" + l32).str();
        auto fold = [&](mlir::Value bDStrip, mlir::Value bMStrip,
                        mlir::Value sumiStrip, mlir::Value sumfStrip) {
          // Scale term (d_x*d_y)*sumi.
          mlir::Value dC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bDStrip, aD, vl8}, opName, role);
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumiStrip, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfStrip)
                  .getResult();
          mlir::Value scaled =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                             mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                             role);
          // MIN term m_x*s_y, added LANE-WISE.
          mlir::Value mC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bMStrip, aS, vl8}, opName, role);
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfaddCallee,
                             mlir::ValueRange{scaled, mC, vl8}, opName, role);
          rewriter.create<emitc::AssignOp>(loc, sumfStrip, nextF);
        };
        for (int64_t h = 0; h < numHalves; ++h)
          fold(bD[h], bM[h], sumi[h], sumfVar[h]);
      }

      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      auto storeHalf = [&](mlir::Value sumfVar, int64_t laneOff) {
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
            rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar).getResult();
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
      };
      for (int64_t h = 0; h < numHalves; ++h)
        storeHalf(sumfVar[h], h * half);
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
    mlir::Value zeroLane =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
    mlir::Value resultTok =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                         seedCallee,
                                         mlir::ValueRange{zeroLane, sizeLit(1)})
            .getResult(0);
    valueMap[gemv.getResult()] = resultTok;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ8_0Q8_0(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlRepackGemvQ80Q80Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<weftrvv::GgmlRepackGemvQ80Q80Op>(op))
        gemv = g;
    }
    if (!gemv)
      return rewriter.notifyMatchFailure(scope, "repack-gemv body missing op");

    mlir::Value weightBase = valueMap.lookup(gemv.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemv.getActivationBase());
    mlir::Value output = valueMap.lookup(gemv.getOutput());
    mlir::Value columnCount = valueMap.lookup(gemv.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(gemv, "repack-gemv ABI operand unmapped");

    llvm::StringRef opName = gemv.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires (no fractional LMUL on the
    // pre-ratification generation): the entire chain shifts up one notch i8m1 ->
    // i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip at VLEN=128. Only
    // the type/callee LMUL suffixes change; numHalves, vl, every loop bound and
    // byte offset are driven by half_lanes and stay identical.
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!gemv.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          gemv, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *gemv.getIntegerCoreLmul();
    // The three element-width LMUL rungs the chain anchors on, keyed off the
    // i8 core anchor: 8-bit core, 16-bit product/scale, 32-bit accumulate/f32 fold.
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
    mlir::Type f16m1Type =
        emitc::OpaqueType::get(ctx, ("vfloat16" + l16 + "_t").str());
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type f16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));

    // The 16x1 repacked GEMV block-format structural facts (I4 mirror, pinned by
    // the verifier): QK=32, block_q8_0x16 weight stride 544, block_q8_0
    // activation stride 34, the FULL int8 weight quants at +32, the activation
    // int8 quants at +2, 16 weight rows per group, and the VLEN=128 e16m1 half
    // width 8. There is NO activation interleave (the activation is a plain q8_0
    // stream, one column) and NO nibble split (q8_0 quants are full int8 -- 32
    // contraction positions per block, one int8 weight byte each).
    int64_t qk = gemv.getQk();
    int64_t weightStride = gemv.getWeightBlockStride();
    int64_t activationStride = gemv.getActivationBlockStride();
    int64_t weightQuantOffset = gemv.getWeightQuantByteOffset();
    int64_t activationQuantOffset = gemv.getActivationQuantByteOffset();
    int64_t weightInterleave = gemv.getWeightInterleave();   // 16
    int64_t half = gemv.getHalfLanes();                      // 8 @128, 16 @256
    // The number of disjoint strips that tile the 16-block-as-lane group is
    // weight_interleave / half_lanes: 16/8 = 2 strips of 8 lanes at VLEN=128, or
    // 16/16 = 1 strip of 16 lanes at VLEN=256. Strip h covers rows
    // [h*half, h*half+half); every strip reads BYTE-IDENTICAL repacked data (the
    // repack is 16-way interleaved: byte i = block(i%16) offset(i/16), so a
    // 16-lane strip at VLEN=256 covers exactly the two 8-lane halves of VLEN=128).
    // The verifier pins half_lanes in {8,16} dividing 16, fail-closed (I7).
    int64_t numHalves = weightInterleave / half;             // 2 @128, 1 @256
    // The contraction has QK=32 positions per block (one int8 weight byte each),
    // NOT the q4_0 nibble-byte count of 16. Every position reads ALL 16 lanes of
    // a strip from the 512 interleaved weight bytes (i in [0,32), byte = 32 +
    // i*16 + h*half), so all 512 weight bytes are read exactly once.
    int64_t contractionPositions = qk;                       // 32

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemv, "repack-gemv output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width (8 i16 lanes at
    // VLEN=128, 16 at VLEN=256). Every intrinsic in the kernel runs at this fixed
    // vl (the patch passes the literal half to every intrinsic).
    mlir::Value vl8 = sizeLit(half);

    // size_t nb = n / QK;  (the contraction block count, shared across groups).
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    // size_t nc_groups = nc / 16;
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // A typed i8 contiguous sub-load: __riscv_vle8_v_i8<l8>((int8_t*)ptr, vl).
    // The repacked q8_0 weight bytes are FULL int8 -- no decode, the loaded lane
    // feeds vwmul directly.
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
    auto loadWeightStrip = [&](mlir::Value base, mlir::Value byteOff)
        -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // A scalar i8 read of the plain activation quant byte a_ptr[l].qs[k]:
    // *(const int8_t *)(ab + 2 + k).
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
    // vwmul_vx widening multiply: prod = act_scalar * weight_strip (i8 -> i16).
    // Full int8 products [-16129, 16256] do NOT fit the q4_0 i16 accumulator, so
    // the product is kept SEPARATE (i16) and folded into an i32 accumulator below.
    std::string vwmulCallee = ("__riscv_vwmul_vx_i16" + l16).str();
    auto vwmul = [&](mlir::Value scalar, mlir::Value vec) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i16m1Type, vwmulCallee,
                            mlir::ValueRange{vec, scalar, vl8}, opName, role);
    };
    // vwadd_wv widening add wide+narrow: acc(i32) += widened prod(i16). Full int8
    // products overflow i16 after 3 terms (127*127*3 > 32767), so accumulation is
    // i32 IN-BLOCK -- this REPLACES q4_0's i16-vwmacc + end-of-block vwadd_vv
    // combine. Integer accumulation is order-independent so the dot is byte-exact.
    std::string vwaddwCallee = ("__riscv_vwadd_wv_i32" + l32).str();
    auto vwaddw = [&](mlir::Value acc, mlir::Value prod) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddwCallee,
                            mlir::ValueRange{acc, prod, vl8}, opName, role);
    };

    // The plain q8_0 activation base is set ONCE (vy) and indexed a[l] -- it is
    // reused across every weight column group x (NO per-group advance, unlike
    // the GEMM's y*nb*stride row-group base).
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

      // const uint8_t *b = vx + x*nb*544;  (the q8_0x16 column group base).
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      mlir::Value bGroup = rewriter.create<emitc::AddOp>(
          loc, weightPtrType, weightBase, bGroupOff);

      // vfloat32m2_t sumf_a = vfmv_v_f(0,8) (rows 0..7), sumf_b (rows 8..15).
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
      // Mutable f32 accumulator lvalues, ONE per strip (the inner block loop
      // carries them). num_halves strips: 2 (rows 0..7, 8..15) at half=8, or 1
      // (rows 0..15) at half=16.
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

        // const uint8_t *bl = b + l*544;   const uint8_t *al = a + l*34;
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

        // vint32m2_t sumi_{a,b} = vmv_v_x(0, 8);  ONE i32 accumulator per strip
        // (full int8 products overflow i16, so the in-block accumulator is i32).
        std::string mvCallee = riscvIntrinsicName("vmv_v_x", 32, l32, "i32");
        auto seedI32 = [&]() -> mlir::Value {
          return emitOpaqueCallBuilt(
              rewriter, loc, i32m2Type, mvCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zero =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {zero, vl8};
              });
        };
        // Per-strip i32 accumulator lvalues, one per strip: at num_halves=2 this
        // is sumi_a then sumi_b.
        llvm::SmallVector<mlir::Value> sumiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto v = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, v, seedI32());
          sumiVar.push_back(v);
        }

        // ===== Contraction-position loop: for (i = 0; i < 32; ++i) =====
        auto posLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(contractionPositions), sizeLit(1),
            /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard ng(rewriter);
          rewriter.setInsertionPointToStart(posLoop.getBody());
          mlir::Value i = posLoop.getInductionVar();

          // num_halves disjoint contiguous sub-loads, strip h at qs[i*16+h*half]:
          // rows 0..7 at qs[i*16+0] and rows 8..15 at qs[i*16+8] (half=8, 2
          // strips), or rows 0..15 at qs[i*16+0] (half=16, 1 strip). byte =
          // 32 + i*16 (+ h*half). LOAD phase first (FileCheck pins both vle8
          // before any product); the h=0 offset is qsOff with NO AddOp.
          step("weight_quant_addr");
          mlir::Value i16 = rewriter.create<emitc::MulOp>(
              loc, sizeType, i, sizeLit(weightInterleave));
          mlir::Value qsOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQuantOffset), i16);
          // Compute every strip's byte offset FIRST, then issue all loads. The
          // h=0 offset is qsOff with NO AddOp.
          llvm::SmallVector<mlir::Value> wByteOff;
          for (int64_t h = 0; h < numHalves; ++h) {
            if (h == 0)
              wByteOff.push_back(qsOff);
            else
              wByteOff.push_back(rewriter.create<emitc::AddOp>(
                  loc, sizeType, qsOff, sizeLit(h * half)));
          }
          llvm::SmallVector<mlir::Value> wStrip;
          for (int64_t h = 0; h < numHalves; ++h)
            wStrip.push_back(loadWeightStrip(bl, wByteOff[h]));

          // Single activation column (SHARED across strips, read ONCE -- this is
          // a GEMV, one column): the int8 quant al.qs[i] at byte = 2 + i. q8_0
          // has NO lo/hi split (full int8, one quant per position).
          step("act_quant_addr");
          mlir::Value aOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset), i);
          mlir::Value aQuant = i8Read(al, aOff);

          // ACCUMULATE phase: per strip, lane-wise vwmul (i8xi8 -> i16 product)
          // then vwadd_wv (i32_acc += widened product). At half=8 this is strip
          // a then strip b.
          for (int64_t h = 0; h < numHalves; ++h) {
            // prod = vwmul(al.qs[i], w_h, vl);  (i8 x i8 -> i16)
            mlir::Value prod = vwmul(aQuant, wStrip[h]);
            // sumi_h = vwadd_wv(sumi_h, prod, vl);  (i32 += widened i16)
            mlir::Value cur =
                rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiVar[h],
                                             vwaddw(cur, prod));
          }
        }

        // The i32 accumulators are ALREADY the per-strip integer dots (q8_0 has
        // no lo/hi combine -- the in-block i32 vwadd_wv already did the widening
        // q4_0 deferred to the end-of-block vwadd_vv). Load them for the fold.
        llvm::SmallVector<mlir::Value> sumi;
        for (int64_t h = 0; h < numHalves; ++h)
          sumi.push_back(
              rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[h])
                  .getResult());

        // vfloat16m1_t b_d_h = vle16(&bl.d[h*half], vl);  one scale strip each.
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
        llvm::SmallVector<mlir::Value> bD;
        for (int64_t h = 0; h < numHalves; ++h)
          bD.push_back(loadScales(h * half));

        // The single activation scale *(const _Float16 *)&al.d (NO float cast),
        // broadcast into both halves' vfwmul.
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

        // d_{a,b} = vfwmul_vf(b_d_{a,b}, aD, 8);
        // sumf_{a,b} = vfmacc_vv(sumf_{a,b}, vfcvt_f_x_v(sumi_{a,b},8), d, 8);
        std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        auto fold = [&](mlir::Value bD, mlir::Value sumi,
                        mlir::Value sumfVar) {
          mlir::Value dC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bD, aD, vl8}, opName, role);
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumi, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar)
                  .getResult();
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                             mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                             role);
          rewriter.create<emitc::AssignOp>(loc, sumfVar, nextF);
        };
        for (int64_t h = 0; h < numHalves; ++h)
          fold(bD[h], sumi[h], sumfVar[h]);
      }

      // Per-strip store vse32(s + x*16 + h*half, sumf_h, vl): at half=8 this is
      // s+x*16+0 (rows 0..7) and s+x*16+8 (rows 8..15), at half=16 one 16-lane
      // store s+x*16+0 (rows 0..15). The laneOff==0 guard keeps the first store
      // AddOp-free.
      std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
      auto storeHalf = [&](mlir::Value sumfVar, int64_t laneOff) {
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
            rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar).getResult();
        emitOpaqueCallVoid(rewriter, loc, vseCallee,
                           mlir::ValueRange{dst, sumfVal, vl8}, opName, role);
      };
      for (int64_t h = 0; h < numHalves; ++h)
        storeHalf(sumfVar[h], h * half);
    }
    if (mlir::failed(status))
      return mlir::failure();

    (void)outPtr;
    // The op result is the typed i32m1 token; the GEMV writes through *s so the
    // token has no consumer. Seed it with a zero i32m1 lane-0 so the valueMap
    // entry is well-formed (mirrors the GEMM's unused result).
    std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
    mlir::Value zeroLane =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
    mlir::Value resultTok =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                         seedCallee,
                                         mlir::ValueRange{zeroLane, sizeLit(1)})
            .getResult(0);
    valueMap[gemv.getResult()] = resultTok;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ4_1Q8_1(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlRepackGemvQ41Q81Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<weftrvv::GgmlRepackGemvQ41Q81Op>(op))
        gemv = g;
    }
    if (!gemv)
      return rewriter.notifyMatchFailure(scope, "repack-gemv-q4_1 body missing op");

    mlir::Value weightBase = valueMap.lookup(gemv.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemv.getActivationBase());
    mlir::Value output = valueMap.lookup(gemv.getOutput());
    mlir::Value columnCount = valueMap.lookup(gemv.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(gemv,
                                         "repack-gemv-q4_1 ABI operand unmapped");

    llvm::StringRef opName = gemv.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one
    // notch i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
    // Fail-closed final capability-fact read: the
    // front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
    if (!gemv.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          gemv, "repack integer core requires an explicit integer_core_lmul "
                "capability fact (front door stamps it; no silent mf2 default)");
    llvm::StringRef coreLmul = *gemv.getIntegerCoreLmul();
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

    // The 16x1 repacked q4_1 GEMV block-format structural facts (I4 mirror,
    // pinned by the verifier): QK=32, block_q4_1x16 weight stride 320 (16 d + 16
    // m + 256 nibble bytes), block_q8_1 activation stride 36, the weight nibble
    // bytes at +64, the per-row MIN strip at +32, the activation int8 quants at
    // +4, the activation scaled-sum at +2, 16 weight rows per group, and the
    // VLEN-derived e16m1 half width.
    int64_t qk = gemv.getQk();
    int64_t weightStride = gemv.getWeightBlockStride();
    int64_t activationStride = gemv.getActivationBlockStride();
    int64_t weightQuantOffset = gemv.getWeightQuantByteOffset();   // 64
    int64_t activationQuantOffset = gemv.getActivationQuantByteOffset(); // 4
    int64_t weightMinOffset = gemv.getWeightMinByteOffset();       // 32
    int64_t activationSumOffset = gemv.getActivationSumByteOffset(); // 2
    int64_t weightInterleave = gemv.getWeightInterleave();   // 16
    int64_t half = gemv.getHalfLanes();                      // 8 @128, 16 @256
    int64_t numHalves = weightInterleave / half;             // 2 @128, 1 @256
    int64_t nibbleBytes = qk / 2;                            // 16 nibble bytes
    int64_t activationHighRow = nibbleBytes;                 // 16

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemv, "repack-gemv-q4_1 output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width.
    mlir::Value vl8 = sizeLit(half);

    // size_t nb = n / QK;
    step("block_count");
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
    // size_t nc_groups = nc / 16;
    step("col_group_count");
    mlir::Value ncGroups = rewriter.create<emitc::DivOp>(
        loc, sizeType, columnCount, sizeLit(weightInterleave));

    // A typed UNSIGNED u8 contiguous sub-load of the raw repacked nibbles:
    // __riscv_vle8_v_u8<l8>((uint8_t*)ptr, vl). q4_1 stores RAW nibbles (NOT the
    // offset-binary biased bytes the q4_0 repack stores).
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

      // const uint8_t *b = vx + x*nb*320;  (the q4_1x16 column group base).
      step("weight_group_base");
      mlir::Value bGroupBlocks =
          rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
      mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
          loc, sizeType, bGroupBlocks, sizeLit(weightStride));
      mlir::Value bGroup = rewriter.create<emitc::AddOp>(
          loc, weightPtrType, weightBase, bGroupOff);

      // vfloat32m2_t sumf_h = vfmv_v_f(0, half) per strip.
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

        // const uint8_t *bl = b + l*320;   const uint8_t *al = a + l*36;
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

        // vint16m1_t sumi_h_{lo,hi} = vmv_v_x(0, half);
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
        llvm::SmallVector<mlir::Value> sumiLoVar, sumiHiVar;
        for (int64_t h = 0; h < numHalves; ++h) {
          auto vlo = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i16m1Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, vlo, seedI16());
          sumiLoVar.push_back(vlo);
          auto vhi = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i16m1Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, vhi, seedI16());
          sumiHiVar.push_back(vhi);
        }

        // ===== Nibble-step loop: for (i = 0; i < 16; ++i) =====
        auto nibLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(nibbleBytes), sizeLit(1),
            /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard ng(rewriter);
          rewriter.setInsertionPointToStart(nibLoop.getBody());
          mlir::Value i = nibLoop.getInductionVar();

          // num_halves disjoint contiguous sub-loads, strip h at qs[i*16+h*half],
          // byte = 64 + i*16 (+ h*half).
          step("weight_nibble_addr");
          mlir::Value i16 = rewriter.create<emitc::MulOp>(
              loc, sizeType, i, sizeLit(weightInterleave));
          mlir::Value qsOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(weightQuantOffset), i16);
          llvm::SmallVector<mlir::Value> wByteOff;
          for (int64_t h = 0; h < numHalves; ++h) {
            if (h == 0)
              wByteOff.push_back(qsOff);
            else
              wByteOff.push_back(rewriter.create<emitc::AddOp>(
                  loc, sizeType, qsOff, sizeLit(h * half)));
          }
          llvm::SmallVector<mlir::Value> packed;
          for (int64_t h = 0; h < numHalves; ++h)
            packed.push_back(loadNibbles(bl, wByteOff[h]));
          // DECODE phase: per strip, lo then hi (UNSIGNED, NO vxor / sign-extend).
          llvm::SmallVector<mlir::Value> bLo, bHi;
          for (int64_t h = 0; h < numHalves; ++h) {
            bLo.push_back(decodeLo(packed[h]));
            bHi.push_back(decodeHi(packed[h]));
          }

          // Single activation column (SHARED across strips, read ONCE): low quant
          // al.qs[i] at +4+i, high quant al.qs[16+i] at +4+16+i.
          step("act_quant_addr_lo");
          mlir::Value loOff = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset), i);
          mlir::Value aLo = i8Read(al, loOff);
          step("act_quant_addr_hi");
          mlir::Value hiBase = rewriter.create<emitc::AddOp>(
              loc, sizeType, sizeLit(activationQuantOffset),
              sizeLit(activationHighRow));
          mlir::Value hiOff =
              rewriter.create<emitc::AddOp>(loc, sizeType, hiBase, i);
          mlir::Value aHi = i8Read(al, hiOff);

          // ACCUMULATE phase: per strip, lane-wise vwmacc lo then hi.
          for (int64_t h = 0; h < numHalves; ++h) {
            mlir::Value curLo =
                rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiLoVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiLoVar[h],
                                             vwmacc(curLo, aLo, bLo[h]));
            mlir::Value curHi =
                rewriter.create<emitc::LoadOp>(loc, i16m1Type, sumiHiVar[h])
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiHiVar[h],
                                             vwmacc(curHi, aHi, bHi[h]));
          }
        }

        // const vint32m2_t sumi_h = vwadd_vv(sumi_h_lo, sumi_h_hi, half);
        std::string vwaddCallee = ("__riscv_vwadd_vv_i32" + l32).str();
        auto combine = [&](mlir::Value loVar, mlir::Value hiVar) -> mlir::Value {
          mlir::Value lo =
              rewriter.create<emitc::LoadOp>(loc, i16m1Type, loVar).getResult();
          mlir::Value hi =
              rewriter.create<emitc::LoadOp>(loc, i16m1Type, hiVar).getResult();
          return emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                                mlir::ValueRange{lo, hi, vl8}, opName, role);
        };
        llvm::SmallVector<mlir::Value> sumi;
        for (int64_t h = 0; h < numHalves; ++h)
          sumi.push_back(combine(sumiLoVar[h], sumiHiVar[h]));

        // vfloat16m1_t b_d_h = vle16(&bl.d[h*half], half);  one delta strip each.
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
        // Per-row delta d strips (at byte 0 + h*half*2) and MIN m strips (at byte
        // weightMinOffset + h*half*2).
        llvm::SmallVector<mlir::Value> bD, bM;
        for (int64_t h = 0; h < numHalves; ++h)
          bD.push_back(loadF16Strip(0, h * half));
        for (int64_t h = 0; h < numHalves; ++h)
          bM.push_back(loadF16Strip(weightMinOffset, h * half));

        // The single activation scale d_y = *(const _Float16 *)&al.d, broadcast
        // into every strip's vfwmul.
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
        // The single activation scaled-sum s_y = *(const _Float16 *)&al.s (at
        // +2), folded into the LANE-WISE MIN term sumf += s_y * m_x. This is the
        // Family-B distinction from q4_0: the MIN correction term.
        mlir::Value aS = emitOpaqueCallBuilt(
            rewriter, loc, f16ScalarType, f16ReadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aSFull = rewriter.create<emitc::AddOp>(
                  loc, activationPtrType, al, sizeLit(activationSumOffset));
              mlir::Value aSCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, aSFull)
                      .getResult();
              return {aSCast};
            },
            llvm::StringRef("act_sum_scalar"));

        // Per strip, ggml's q4_1 fold sumf += (d_x*d_y)*sumi + m_x*s_y:
        //   d_h  = vfwmul_vf(b_d_h, d_y, half);          // (d_x * d_y)
        //   sumf_h = vfmacc_vv(sumf_h, vfcvt(sumi_h), d_h);  // + (d_x*d_y)*sumi
        //   m_h  = vfwmul_vf(b_m_h, s_y, half);          // (m_x * s_y)
        //   sumf_h = vfadd_vv(sumf_h, m_h, half);        // + m_x*s_y
        std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        std::string vfaddCallee = ("__riscv_vfadd_vv_f32" + l32).str();
        auto fold = [&](mlir::Value bDStrip, mlir::Value bMStrip,
                        mlir::Value sumiStrip, mlir::Value sumfStrip) {
          // Scale term (d_x*d_y)*sumi.
          mlir::Value dC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bDStrip, aD, vl8}, opName, role);
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                             mlir::ValueRange{sumiStrip, vl8}, opName, role);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfStrip)
                  .getResult();
          mlir::Value scaled =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                             mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                             role);
          // MIN term m_x*s_y, added LANE-WISE.
          mlir::Value mC =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                             mlir::ValueRange{bMStrip, aS, vl8}, opName, role);
          mlir::Value nextF =
              emitOpaqueCall(rewriter, loc, f32m2Type, vfaddCallee,
                             mlir::ValueRange{scaled, mC, vl8}, opName, role);
          rewriter.create<emitc::AssignOp>(loc, sumfStrip, nextF);
        };
        for (int64_t h = 0; h < numHalves; ++h)
          fold(bD[h], bM[h], sumi[h], sumfVar[h]);
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
    // The op result is the typed i32m1 token; the GEMV writes through *s so the
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
    valueMap[gemv.getResult()] = resultTok;
    return mlir::success();
  }


} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
