#include "RVVToEmitCInternal.h"
#include "TianChenRV/Conversion/RVV/RVVToEmitCSupport.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

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

namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace detail {

// VariantToEmitCFunc block-quant linear (block-dot / GEMM) emit methods:
// q4_0/q8_0, q4_1/q5_0/q5_1, q1_0 + the q4_0 gemm-tile / gemm tilings. Split out
// of RVVToEmitC.cpp as a pure code move; the emitted C is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitQ4_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: q4_0 offset_binary_nibble / half-block / LeftAssoc instance of the descriptor-driven emitFlatBlockDot.
    // Resolve the ABI operands + provenance, derive the block-format descriptor
    // (from `kind`/attrs) + the scheduled BlockDotFacts, emit the shared body.
    tcrvrvv::GgmlBlockDotQ40Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ40Q80Op>(op))
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
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-plain");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ5_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: q5_0 five_bit_offset_binary (qh) / half-block / ScalesTimesSumi instance of the descriptor-driven emitFlatBlockDot.
    // Resolve the ABI operands + provenance, derive the block-format descriptor
    // (from `kind`/attrs) + the scheduled BlockDotFacts, emit the shared body.
    tcrvrvv::GgmlBlockDotQ50Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ50Q80Op>(op))
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
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-plain");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ5_1Q8_1BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: q5_1 five_bit_offset_binary (qh, no bias) / half-block / ScalePlusMin instance of the descriptor-driven emitFlatBlockDot.
    // Resolve the ABI operands + provenance, derive the block-format descriptor
    // (from `kind`/attrs) + the scheduled BlockDotFacts, emit the shared body.
    tcrvrvv::GgmlBlockDotQ51Q81Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ51Q81Op>(op))
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
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-plain");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_0Q8_0GemmTile(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlGemmTileQ40Q80Op tile;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto t = llvm::dyn_cast<tcrvrvv::GgmlGemmTileQ40Q80Op>(op))
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

    llvm::StringRef opName = tile.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = tile.getTCRVEmitCLowerableSourceRole();
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
    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlGemmQ40Q80Op gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlGemmQ40Q80Op>(op))
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

    llvm::StringRef opName = gemm.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getTCRVEmitCLowerableSourceRole();
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
    // M (the inner activation-column block) is the G3 measurement-tuned knob: the
    // materialize pass stamps the measured-best M, and absent a pass run the op
    // falls back to the default cache-friendly tile (M=4, the measured rv64gcv
    // winner -- INC-25 G2: M=4 ~1.04x, M=6 ~0.857x regression). The attribute is
    // OPTIONAL so a raw op stays lowerable; the literal mirrors the plugin's
    // kRVVGemmDefaultActivationCols (the conversion layer stays free of the plugin
    // schedule header -- it is the consumer of the stamped attr, not the authority).
    constexpr int64_t kGemmDefaultActivationCols = 4;
    int64_t cols = gemm.getActivationCols().value_or(kGemmDefaultActivationCols);

    // The weight decode anchors at the m1 whole-half-block form (one
    // vsetvl_e8m1(16) covers the 16 nibble bytes at VLEN >= 128); the product
    // widens i8m1 -> i16m2. The *how* (vector grouping), never the *what*: the
    // dot product is byte-exact (vwredsum sums the same integer set).
    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmQ4_0Q8_0(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemmQ40Q80Op gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemmQ40Q80Op>(op))
        gemm = g;
    }
    if (!gemm)
      return rewriter.notifyMatchFailure(scope, "repack-gemm body missing op");

    mlir::Value weightBase = valueMap.lookup(gemm.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemm.getActivationBase());
    mlir::Value output = valueMap.lookup(gemm.getOutput());
    mlir::Value rowCount = valueMap.lookup(gemm.getRowCount());
    mlir::Value columnCount = valueMap.lookup(gemm.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(gemm.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(gemm, "repack-gemm ABI operand unmapped");

    llvm::StringRef opName = gemm.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getTCRVEmitCLowerableSourceRole();
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
    llvm::StringRef coreLmul = gemm.getIntegerCoreLmul().value_or("mf2");
    // The three element-width LMUL rungs the chain anchors on, keyed off the
    // i8 core anchor: 8-bit core, 16-bit product/scale, 32-bit combine/f32 fold.
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

    // The 16x1 repacked block-format structural facts (I4 mirror, pinned by the
    // verifier): QK=32, block_q4_0x16 stride 288, block_q8_0x4 stride 136, the
    // weight nibble bytes at +32, the activation int8 quants at +8, 16 weight
    // rows / 4 activation columns per group, and the VLEN=128 e16m1 half width 8.
    int64_t qk = gemm.getQk();
    int64_t weightStride = gemm.getWeightBlockStride();
    int64_t activationStride = gemm.getActivationBlockStride();
    int64_t weightQuantOffset = gemm.getWeightQuantByteOffset();
    int64_t activationQuantOffset = gemm.getActivationQuantByteOffset();
    int64_t weightInterleave = gemm.getWeightInterleave();   // 16
    int64_t activationInterleave = gemm.getActivationInterleave(); // 4
    int64_t half = gemm.getHalfLanes();                      // 8 @128, 16 @256
    // The number of disjoint strips that tile the 16-block-as-lane group is
    // weight_interleave / half_lanes: 16/8 = 2 strips of 8 lanes at VLEN=128, or
    // 16/16 = 1 strip of 16 lanes at VLEN=256. This is the ONLY thing that
    // diverges with VLEN -- every strip reads BYTE-IDENTICAL repacked data (the
    // repack is 16-way interleaved: byte i = block(i%16) offset(i/16), so a
    // 16-lane strip at VLEN=256 covers exactly the two 8-lane halves of VLEN=128).
    // The verifier pins half_lanes in {8,16} dividing 16, fail-closed (I7).
    int64_t numHalves = weightInterleave / half;             // 2 @128, 1 @256
    // The number of activation columns folded per pass over the contraction
    // block loop. The RVV1.0 fractional chain (i8mf2 -> i16m1 -> i32m2 -> f32m2)
    // holds all 4 columns at once -- 4 f32m2 accumulators + 8 i16m1 products =
    // 16 vregs, well within the 32-register file -- so it folds all 4 in ONE
    // pass (columnsPerPass = activationInterleave), byte-identical to before.
    // The RVV0.7.1 whole-LMUL chain (i8m1 -> i16m2 -> i32m4 -> f32m4) DOUBLES
    // every rung: 4 f32m4 accumulators (16 vregs) that must persist across the
    // block loop, PLUS 8 i16m2 products (16 vregs) live during the nibble loop
    // = 32 vregs of state before any load/temp -> forced spill. GCC's spill of
    // the last (4th) f32m4 accumulator mis-vtypes the zero-seed store to e16,m2
    // (it zeros only lanes 0-7; lanes 8-15 keep stack garbage), corrupting the
    // 4th column's upper lanes (observed: worst error always row=3, lane>=8).
    // Folding ONE column per pass makes the per-pass live set IDENTICAL to the
    // (bit-exact) GEMV: 1 f32m4 accumulator (4 vregs) + 1 i16m2 lo + 1 i16m2 hi
    // (2 vregs each) + nibble temps ~= 12-14 vregs peak, well under 32 -> GCC
    // keeps the accumulator in a vreg across the whole block loop, never spilling
    // it. (A 2-column split still left 2 f32m4 + 4 i16m2 = 16 vregs of carried
    // state -- enough for GCC to still spill one accumulator under the e16,m2
    // nibble-loop vtype, producing the SAME mis-vtyped e32,m4 -> e16,m2 store,
    // just relocated to the 2nd column; objdump confirmed it.) The cost is
    // re-decoding the shared weight nibbles once per column (correctness-first;
    // this is the prefill GEMM -- the hot decode path is the already-bit-exact
    // GEMV); the whole chain stays whole-LMUL, no fractional symbol. RVV1.0 takes
    // the else branch (one pass over all 4 columns), emission byte-identical.
    int64_t columnsPerPass =
        (coreLmul == "m1") ? 1 : activationInterleave;       // 1 @rvv07; 4 @rvv1.0
    int64_t nibbleBytes = qk / 2;                            // 16 nibble bytes
    // The activation high-half int8 quants start after the 32 low-half quants
    // (the 4 columns x 8 lanes the low half consumes per nibble step). The patch
    // reads a_ptr[l].qs[64 + i*4 + c]; 64 = activationInterleave * nibbleBytes.
    int64_t activationHighRow = activationInterleave * nibbleBytes; // 64

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemm, "repack-gemm output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width (8 i16 lanes at
    // VLEN=128). Every intrinsic in the kernel runs at this fixed vl (the patch
    // passes the literal 8 to every intrinsic).
    mlir::Value vl8 = sizeLit(half);

    // size_t nb = n / QK;  (the contraction block count, shared across groups).
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

    // A typed i8 contiguous sub-load: __riscv_vle8_v_i8<l8>((int8_t*)ptr, vl).
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
    auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // The repacked nibbles already carry the ^0x88 offset-binary bias, so the
    // decode is a plain sign-extension: b_lo = vsra(vsll(b,4),4); b_hi=vsra(b,4).
    std::string sllCallee = ("__riscv_vsll_vx_i8" + l8).str();
    std::string sraCallee = ("__riscv_vsra_vx_i8" + l8).str();
    mlir::Value four = sizeLit(4);
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      mlir::Value shl =
          emitOpaqueCall(rewriter, loc, i8mf2Type, sllCallee,
                         mlir::ValueRange{packed, four, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                            mlir::ValueRange{shl, four, vl8}, opName, role);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                            mlir::ValueRange{packed, four, vl8}, opName, role);
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

      // const uint8_t *a = vy + y*nb*136;  (the q8_0x4 row group base).
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

        // const uint8_t *b = vx + x*nb*288;  (the q4_0x16 column group base).
        step("weight_group_base");
        mlir::Value bGroupBlocks =
            rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
        mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, bGroupBlocks, sizeLit(weightStride));
        mlir::Value bGroup = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, weightBase, bGroupOff);

        // ===== Strip loop: for (h = 0; h < num_halves; ++h) {roff = h*half} =====
        // num_halves = weight_interleave/half_lanes (2 strips of 8 @128, 1 strip
        // of 16 @256); vl and roff already scale with `half`, so a 16-lane strip
        // runs the body ONCE at vl=16, roff=0.
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
          // RVV1.0 (columnsPerPass=4) runs ONE pass over c in [0,4) -- emission
          // byte-identical to the un-split form. RVV0.7 (columnsPerPass=1) runs
          // FOUR passes (c in [0,1),[1,2),[2,3),[3,4)), each re-decoding the
          // shared weight nibbles but holding only 1 f32m4 accumulator live ->
          // per-pass live set == the bit-exact GEMV -> no spill (the 2-column
          // split still spilled; see the columnsPerPass rationale above).
          for (int64_t cLo = 0; cLo < activationInterleave;
               cLo += columnsPerPass) {
          int64_t cHi = cLo + columnsPerPass;
          // vfloat32m2_t sumf_{cLo..cHi} = vfmv_v_f(0.0f, 8);  (per-pass f32 acc)
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
          // Mutable f32 accumulator lvalues (the inner block loop carries them).
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

            // const uint8_t *bl = b + l*288;   const uint8_t *al = a + l*136;
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

            // vint16m1_t sumi_{0..3}_{lo,hi} = vmv_v_x(0, 8);
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

              // b_packed = vle8(&bl.qs[i*16 + roff], 8);  byte = 32 + i*16 + roff
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
                // sumi_c_lo = vwmacc_vx(sumi_c_lo, al.qs[i*4+c], b_lo, 8);
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

                // sumi_c_hi = vwmacc_vx(sumi_c_hi, al.qs[64+i*4+c], b_hi, 8);
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
              sumi32[c] =
                  emitOpaqueCall(rewriter, loc, i32m2Type, vwaddCallee,
                                 mlir::ValueRange{lo, hi, vl8}, opName, role);
            }

            // vfloat16m1_t b_d = vle16(&bl.d[roff], 8);  byte = roff*2.
            step("weight_scale_addr");
            mlir::Value dByteOff = rewriter.create<emitc::MulOp>(
                loc, sizeType, roff, sizeLit(2));
            mlir::Value dFull = rewriter.create<emitc::AddOp>(
                loc, weightPtrType, bl, dByteOff);
            mlir::Value dCast =
                rewriter.create<emitc::CastOp>(loc, f16PtrType, dFull)
                    .getResult();
            std::string f16LoadCallee = riscvIntrinsicName("vle", 16, l16, "f16");
            mlir::Value bD =
                emitOpaqueCall(rewriter, loc, f16m1Type, f16LoadCallee,
                               mlir::ValueRange{dCast, vl8}, opName, role);

            // d_c = vfwmul_vf(b_d, *(const _Float16 *)&al.d[c], 8);  -- the raw
            // _Float16 activation scale (NO float cast).
            std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
            std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32,
                                                         "f32");
            std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
            llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
            for (int64_t c = cLo; c < cHi; ++c) {
              mlir::Type f16ScalarType =
                  emitc::OpaqueType::get(ctx, "_Float16");
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
              mlir::Value dC =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfwmulCallee,
                                 mlir::ValueRange{bD, aD, vl8}, opName, role);
              // sumf_c = vfmacc_vv(sumf_c, vfcvt_f_x_v(sumi_c, 8), d_c, 8);
              mlir::Value sumiF =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfcvtCallee,
                                 mlir::ValueRange{sumi32[c], vl8}, opName, role);
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                      .getResult();
              mlir::Value nextF =
                  emitOpaqueCall(rewriter, loc, f32m2Type, vfmaccCallee,
                                 mlir::ValueRange{curF, sumiF, dC, vl8}, opName,
                                 role);
              rewriter.create<emitc::AssignOp>(loc, sumfVar[c], nextF);
            }
          }

          // vse32(s + (y*4 + c)*bs + x*16 + roff, sumf_c, 8);  -- the 4x8 store.
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
    // token has no consumer. Seed it with a zero i32m1 lane-0 so the valueMap
    // entry is well-formed (mirrors the per-row GEMM's unused result).
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

mlir::LogicalResult VariantToEmitCFunc::emitPackQ4_0ToX16(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlPackQ40ToX16Op pack;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto p = llvm::dyn_cast<tcrvrvv::GgmlPackQ40ToX16Op>(op))
        pack = p;
    }
    if (!pack)
      return rewriter.notifyMatchFailure(scope, "pack body missing op");

    mlir::Value src = valueMap.lookup(pack.getSrc());
    mlir::Value dst = valueMap.lookup(pack.getDst());
    mlir::Value nblocks = valueMap.lookup(pack.getNblocks());
    if (!src || !dst || !nblocks)
      return rewriter.notifyMatchFailure(pack, "pack ABI operand unmapped");

    llvm::StringRef opName = pack.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = pack.getTCRVEmitCLowerableSourceRole();

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

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
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

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ4_0Q8_0(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemvQ40Q80Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemvQ40Q80Op>(op))
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

    llvm::StringRef opName = gemv.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getTCRVEmitCLowerableSourceRole();
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
    llvm::StringRef coreLmul = gemv.getIntegerCoreLmul().value_or("mf2");
    // The three element-width LMUL rungs the chain anchors on, keyed off the
    // i8 core anchor: 8-bit core, 16-bit product/scale, 32-bit combine/f32 fold.
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
    // the verifier): QK=32, block_q4_0x16 weight stride 288, block_q8_0
    // activation stride 34, the weight nibble bytes at +32, the activation int8
    // quants at +2, 16 weight rows per group, and the VLEN=128 e16m1 half width
    // 8. There is NO activation interleave (the activation is a plain q8_0
    // stream, one column).
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
    int64_t nibbleBytes = qk / 2;                            // 16 nibble bytes
    // The activation high-half int8 quants start after the 16 low-half quants
    // (the single column's first half-block). The patch reads a_ptr[l].qs[16+i].
    int64_t activationHighRow = nibbleBytes;                 // 16

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemv, "repack-gemv output not pointer");
    auto outPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(output);

    // The active vl is the COMPILE-TIME-CONSTANT half width (8 i16 lanes at
    // VLEN=128). Every intrinsic in the kernel runs at this fixed vl (the patch
    // passes the literal 8 to every intrinsic).
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
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, l8, "i8");
    auto loadNibbles = [&](mlir::Value base, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, base, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      return emitOpaqueCall(rewriter, loc, i8mf2Type, i8LoadCallee,
                            mlir::ValueRange{cast, vl8}, opName, role);
    };
    // The repacked nibbles already carry the ^0x88 offset-binary bias, so the
    // decode is a plain sign-extension: b_lo = vsra(vsll(b,4),4); b_hi=vsra(b,4).
    std::string sllCallee = ("__riscv_vsll_vx_i8" + l8).str();
    std::string sraCallee = ("__riscv_vsra_vx_i8" + l8).str();
    mlir::Value four = sizeLit(4);
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      mlir::Value shl =
          emitOpaqueCall(rewriter, loc, i8mf2Type, sllCallee,
                         mlir::ValueRange{packed, four, vl8}, opName, role);
      return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                            mlir::ValueRange{shl, four, vl8}, opName, role);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, i8mf2Type, sraCallee,
                            mlir::ValueRange{packed, four, vl8}, opName, role);
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

      // const uint8_t *b = vx + x*nb*288;  (the q4_0x16 column group base).
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
      // (rows 0..15) at half=16. At half=8 this emits sumf_a then sumf_b in the
      // SAME order as the explicit HEAD code (byte-identical).
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

        // const uint8_t *bl = b + l*288;   const uint8_t *al = a + l*34;
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
          // DECODE phase: per strip, lo then hi (plain sign-extension, NO vxor).
          llvm::SmallVector<mlir::Value> bLo, bHi;
          for (int64_t h = 0; h < numHalves; ++h) {
            bLo.push_back(decodeLo(packed[h]));
            bHi.push_back(decodeHi(packed[h]));
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
      // AddOp-free (byte-identical to HEAD's storeHalf(sumfAVar, 0)).
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
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemvQ50Q80Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemvQ50Q80Op>(op))
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

    llvm::StringRef opName = gemv.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::StringRef coreLmul = gemv.getIntegerCoreLmul().value_or("mf2");
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

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
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

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ8_0Q8_0(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemvQ80Q80Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemvQ80Q80Op>(op))
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

    llvm::StringRef opName = gemv.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getTCRVEmitCLowerableSourceRole();
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
    llvm::StringRef coreLmul = gemv.getIntegerCoreLmul().value_or("mf2");
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

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
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
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemvQ41Q81Op gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemvQ41Q81Op>(op))
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

    llvm::StringRef opName = gemv.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one
    // notch i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
    llvm::StringRef coreLmul = gemv.getIntegerCoreLmul().value_or("mf2");
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

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
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

// q4_K GEVM emitter — 1b-i scaffold (q4_1 GEVM renamed). COMPILES + reads q4_K ABI via shared
// getters, but COMPUTES q4_1-style (single block fold) = NUMERICALLY WRONG until 1b-iii adapts
// it to q4_K's 8-sub-block 6-bit-unpack + bsums-min (see KQUANT-REPACK-DESIGN.md STAGE-1b).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemvQ4KQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemvQ4KQ8KOp gemv;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemvQ4KQ8KOp>(op))
        gemv = g;
    }
    if (!gemv)
      return rewriter.notifyMatchFailure(scope, "repack-gemv-q4_K body missing op");

    mlir::Value weightBase = valueMap.lookup(gemv.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemv.getActivationBase());
    mlir::Value output = valueMap.lookup(gemv.getOutput());
    mlir::Value columnCount = valueMap.lookup(gemv.getColumnCount());
    if (!weightBase || !activationBase || !output || !columnCount)
      return rewriter.notifyMatchFailure(gemv,
                                         "repack-gemv-q4_K ABI operand unmapped");

    llvm::StringRef opName = gemv.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemv.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one
    // notch i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
    llvm::StringRef coreLmul = gemv.getIntegerCoreLmul().value_or("mf2");
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
    // group, 8 sub-blocks of 32, and the VLEN-derived e8 half width.
    int64_t qk = gemv.getQk();                                      // 256
    int64_t weightStride = gemv.getWeightBlockStride();             // 2304
    int64_t activationStride = gemv.getActivationBlockStride();     // 292
    int64_t weightQuantOffset = gemv.getWeightQuantByteOffset();    // 256
    int64_t activationQuantOffset = gemv.getActivationQuantByteOffset(); // 4
    int64_t weightDminOffset = gemv.getWeightDminByteOffset();      // 32
    int64_t weightScalesOffset = gemv.getWeightScalesByteOffset();  // 64
    int64_t activationBsumsOffset = gemv.getActivationBsumsByteOffset(); // 260
    int64_t nSubblocks = gemv.getNSubblocks();                      // 8
    int64_t weightInterleave = gemv.getWeightInterleave();          // 16
    int64_t half = gemv.getHalfLanes();              // 8 @128, 16 @256
    int64_t numHalves = weightInterleave / half;     // 2 @128, 1 @256
    int64_t subBlockSize = qk / nSubblocks;          // 32 elems / sub-block
    int64_t nSuperHalves = qk / 128;                 // 2 (QK_K / 128)
    int64_t subPerSuper = nSubblocks / nSuperHalves; // 4 sub-blocks / super-half
    (void)subBlockSize;

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemv, "repack-gemv-q4_K output not pointer");
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
    // The op result is the typed i32m1 token; the GEVM writes through *s so the
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

mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmQ4_1Q8_1(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemmQ41Q81Op gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemmQ41Q81Op>(op))
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

    llvm::StringRef opName = gemm.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one notch
    // i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip.
    llvm::StringRef coreLmul = gemm.getIntegerCoreLmul().value_or("mf2");
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

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
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
// NUMERIC STATUS: structurally complete, NUMERICALLY UNVERIFIED -- the byte-exact rvv oracle vs
// ggml_gemm_q4_K_16x1_q8_K is a deferred follow-up (see the handoff in the task FINDING).
mlir::LogicalResult VariantToEmitCFunc::emitRepackGemmQ4KQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRepackGemmQ4KQ8KOp gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlRepackGemmQ4KQ8KOp>(op))
        gemm = g;
    }
    if (!gemm)
      return rewriter.notifyMatchFailure(scope,
                                         "repack-gemm-q4_K body missing op");

    mlir::Value weightBase = valueMap.lookup(gemm.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemm.getActivationBase());
    mlir::Value output = valueMap.lookup(gemm.getOutput());
    mlir::Value rowCount = valueMap.lookup(gemm.getRowCount());
    mlir::Value columnCount = valueMap.lookup(gemm.getColumnCount());
    mlir::Value outputRowStride = valueMap.lookup(gemm.getOutputRowStride());
    if (!weightBase || !activationBase || !output || !rowCount ||
        !columnCount || !outputRowStride)
      return rewriter.notifyMatchFailure(gemm,
                                         "repack-gemm-q4_K ABI operand unmapped");

    llvm::StringRef opName = gemm.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // The integer-product core LMUL anchor (the *how*, never the *what*; the
    // 16-way interleaved repack reads the SAME bytes either way). "mf2" (default,
    // absent attribute) is the RVV1.0 fractional chain i8mf2 -> i16m1 -> i32m2 ->
    // f32m2 (f16 scale m1), running at half_lanes e16m1 lanes per strip. "m1" is
    // the WHOLE-LMUL chain RVV0.7.1 requires: the entire chain shifts up one notch
    // i8m1 -> i16m2 -> i32m4 -> f32m4 (f16 scale m2), ONE 16-lane strip. This is
    // the SAME parametric chain the q4_K GEVM sibling carries; the GEMM inherits
    // the already-LMUL-parametric 8-sub-block 6-bit unpack from it.
    llvm::StringRef coreLmul = gemm.getIntegerCoreLmul().value_or("mf2");
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
    int64_t qk = gemm.getQk();                                      // 256
    int64_t weightStride = gemm.getWeightBlockStride();             // 2304
    int64_t activationStride = gemm.getActivationBlockStride();     // 1168
    int64_t weightQuantOffset = gemm.getWeightQuantByteOffset();    // 256
    int64_t activationQuantOffset = gemm.getActivationQuantByteOffset(); // 16
    int64_t weightDminOffset = gemm.getWeightDminByteOffset();      // 32
    int64_t weightScalesOffset = gemm.getWeightScalesByteOffset();  // 64
    int64_t activationBsumsOffset = gemm.getActivationBsumsByteOffset(); // 1040
    int64_t nSubblocks = gemm.getNSubblocks();                      // 8
    int64_t weightInterleave = gemm.getWeightInterleave();          // 16
    int64_t activationInterleave = gemm.getActivationInterleave();  // 4
    int64_t half = gemm.getHalfLanes();              // 8 @128, 16 @256
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

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemm,
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

    // ===== Outer activation-ROW-GROUP loop: for (y = 0; y < nr/4; ++y) =====
    auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nrGroups,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    mlir::LogicalResult status = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard rg(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value y = rowLoop.getInductionVar();

      // const uint8_t *a = vy + y*nb*1168;  (the q8_Kx4 row group base).
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

        // const uint8_t *b = vx + x*nb*2304;  (the q4_Kx16 column group base).
        step("weight_group_base");
        mlir::Value bGroupBlocks =
            rewriter.create<emitc::MulOp>(loc, sizeType, x, nb);
        mlir::Value bGroupOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, bGroupBlocks, sizeLit(weightStride));
        mlir::Value bGroup = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, weightBase, bGroupOff);

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
        // sumfVar[c][h]: per activation column c, per weight strip h.
        llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumfVar(
            activationInterleave);
        for (int64_t c = cLo; c < cHi; ++c) {
          for (int64_t h = 0; h < numHalves; ++h) {
            auto v = rewriter.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32m2Type),
                emitc::OpaqueAttr::get(ctx, ""));
            rewriter.create<emitc::AssignOp>(loc, v, seedF32());
            sumfVar[c].push_back(v);
          }
        }

        // ===== Inner contraction-BLOCK loop: for (l = 0; l < nb; ++l) =====
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

          // -- SHARED weight strips, unpacked ONCE per 16-weight group and reused
          // across the activation columns (the prefill amortization). dmin/d are
          // per-column-lane fp16 strips widened to f32; the per-strip dmin is
          // multiplied by EACH activation column's d_y_c inside the column fold.
          llvm::SmallVector<mlir::Value> dminF32(numHalves), dF32(numHalves);
          for (int64_t h = 0; h < numHalves; ++h) {
            dminF32[h] = widenF16(loadF16Strip(bl, weightDminOffset, h * half));
            dF32[h] = widenF16(loadF16Strip(bl, 0, h * half));
          }

          // ===== Per-column i32 main + bsums accumulators (per strip). =====
          // sumiVar[c][h] (scale main term), bsumsVar[c][h] (min term).
          llvm::SmallVector<llvm::SmallVector<mlir::Value>> sumiVar(
              activationInterleave),
              bsumsVar(activationInterleave);
          for (int64_t c = cLo; c < cHi; ++c) {
            for (int64_t h = 0; h < numHalves; ++h) {
              auto sv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, sv, seedI32());
              sumiVar[c].push_back(sv);
              auto bv = rewriter.create<emitc::VariableOp>(
                  loc, emitc::LValueType::get(i32m2Type),
                  emitc::OpaqueAttr::get(ctx, ""));
              rewriter.create<emitc::AssignOp>(loc, bv, seedI32());
              bsumsVar[c].push_back(bv);
            }
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
            // Unpacked ONCE; scaleVal[h][sb]/minVal[h][sb] reused per column.
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

            // ----- MIN term per activation column m: bsums_acc[m] += bsum_pair *
            // min_sb. The interleaved q8_Kx4 bsums are group16-major/column-minor
            // (index = group16*4 + m); sub-block gsub spans groups 2*gsub and
            // 2*gsub+1, so the paired bsum for column m is
            //   a.bsums[gsub*8 + m] + a.bsums[gsub*8 + m + 4]
            // (the GEVM's gsub*2 / gsub*2+1 single-column pair, x4-interleaved).
            step("min_bsums_fold");
            for (int64_t sb = 0; sb < subPerSuper; ++sb) {
              int64_t gsub = j * subPerSuper + sb;
              for (int64_t c = cLo; c < cHi; ++c) {
                mlir::Value bs0 = i16Read(
                    al, activationBsumsOffset + (gsub * 8 + c) * 2);
                mlir::Value bs1 = i16Read(
                    al, activationBsumsOffset + (gsub * 8 + c + 4) * 2);
                mlir::Value bsPair =
                    rewriter.create<emitc::AddOp>(loc, i32Type, bs0, bs1);
                for (int64_t h = 0; h < numHalves; ++h) {
                  mlir::Value curB =
                      rewriter
                          .create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[c][h])
                          .getResult();
                  rewriter.create<emitc::AssignOp>(
                      loc, bsumsVar[c][h],
                      vwmaccVX32(curB, bsPair, minVal[h][sb]));
                }
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
                // Per-column i16 partials for this 16-element k-chunk.
                llvm::SmallVector<llvm::SmallVector<mlir::Value>> sLo(
                    activationInterleave),
                    sHi(activationInterleave);
                for (int64_t c = cLo; c < cHi; ++c) {
                  for (int64_t h = 0; h < numHalves; ++h) {
                    sLo[c].push_back(seedI16());
                    sHi[c].push_back(seedI16());
                  }
                }
                for (int64_t ii = 0; ii < 16; ++ii) {
                  int64_t i = k * 16 + ii;
                  // SHARED weight nibble decode per (i,h): reused over columns.
                  llvm::SmallVector<mlir::Value> nLo(numHalves), nHi(numHalves);
                  for (int64_t h = 0; h < numHalves; ++h) {
                    step("weight_nibble_addr");
                    mlir::Value packed = loadU8Strip(
                        bl, sizeLit(qsPairBase + i * 16 + h * half));
                    nLo[h] = reinterpretToI8(u8Imm(vandCallee, packed, "0x0F"));
                    nHi[h] = reinterpretToI8(u8Imm(vsrlCallee, packed, "4"));
                  }
                  for (int64_t c = cLo; c < cHi; ++c) {
                    step("act_quant_addr");
                    mlir::Value aLo =
                        i8Read(al, sizeLit(aLoBase + (i * 4 + c)));
                    mlir::Value aHi =
                        i8Read(al, sizeLit(aHiBase + (i * 4 + c)));
                    for (int64_t h = 0; h < numHalves; ++h) {
                      sLo[c][h] = vwmacc16(sLo[c][h], aLo, nLo[h]);
                      sHi[c][h] = vwmacc16(sHi[c][h], aHi, nHi[h]);
                    }
                  }
                }
                // sumi_c += scale_sbLo * sLo_c + scale_sbHi * sHi_c (i16->i32).
                step("scale_subblock_fold");
                for (int64_t c = cLo; c < cHi; ++c) {
                  for (int64_t h = 0; h < numHalves; ++h) {
                    mlir::Value cur0 =
                        rewriter
                            .create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                            .getResult();
                    mlir::Value acc0 =
                        vwmaccVV32(cur0, scaleVal[h][sbLo], sLo[c][h]);
                    rewriter.create<emitc::AssignOp>(
                        loc, sumiVar[c][h],
                        vwmaccVV32(acc0, scaleVal[h][sbHi], sHi[c][h]));
                  }
                }
              }
            }
          }

          // ===== End-of-block fold per column per strip: sumf_c += d_x*d_y_c*sumi
          // (main) then sumf_c -= dmin_x*d_y_c*bsums_c (MIN). d_x/dmin_x are the
          // SHARED per-strip fp16 d/dmin widened to f32; multiplied by the
          // per-column fp32 d_y_c. =====
          for (int64_t c = cLo; c < cHi; ++c) {
            for (int64_t h = 0; h < numHalves; ++h) {
              // d_0_c = dF32[h] * d_y_c;  sumf_c += cvt(sumi_c) * d_0_c.
              mlir::Value d0 = fmulScalar(dF32[h], aD[c]);
              mlir::Value sumiV =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, sumiVar[c][h])
                      .getResult();
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c][h])
                      .getResult();
              mlir::Value afterMain = emitOpaqueCallBuilt(
                  rewriter, loc, f32m2Type, vfmaccVVCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    return {curF, cvtI32F32(sumiV), d0, vl8};
                  });
              // dmin_0_c = dminF32[h] * d_y_c;  sumf_c -= dmin_0_c * cvt(bsums_c).
              mlir::Value dmin0 = fmulScalar(dminF32[h], aD[c]);
              mlir::Value bsumsV =
                  rewriter.create<emitc::LoadOp>(loc, i32m2Type, bsumsVar[c][h])
                      .getResult();
              mlir::Value afterMin = emitOpaqueCallBuilt(
                  rewriter, loc, f32m2Type, vfnmsacVVCallee, opName, role,
                  [&](mlir::OpBuilder &b,
                      mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                    return {afterMain, dmin0, cvtI32F32(bsumsV), vl8};
                  });
              rewriter.create<emitc::AssignOp>(loc, sumfVar[c][h], afterMin);
            }
          }
        }

        // Per-column per-strip store: s + (y*4 + c)*bs + x*16 + h*half.
        std::string vseCallee = riscvIntrinsicName("vse", 32, l32, "f32");
        for (int64_t c = cLo; c < cHi; ++c) {
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
        }
        } // end activation-column-PASS loop (cLo)
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

mlir::LogicalResult VariantToEmitCFunc::emitQ4_1Q8_1BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: q4_1 unsigned_nibble / half-block / ScalePlusMin instance of the descriptor-driven emitFlatBlockDot.
    // Resolve the ABI operands + provenance, derive the block-format descriptor
    // (from `kind`/attrs) + the scheduled BlockDotFacts, emit the shared body.
    tcrvrvv::GgmlBlockDotQ41Q81Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ41Q81Op>(op))
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
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-plain");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ8_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: q8_0 is the plain_i8 / whole-block / SumiTimesScales instance of
    // the descriptor-driven emitFlatBlockDot. Resolve the ABI operands +
    // provenance, derive the block-format descriptor (from `kind`/attrs) + the
    // scheduled BlockDotFacts, and emit the shared flat-plain body.
    tcrvrvv::GgmlBlockDotQ80Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ80Q80Op>(op))
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
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-plain");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

std::optional<FlatBlockDotDescriptor>
deriveFlatBlockDotDescriptor(mlir::Operation *op) {
  auto kindAttr = op->getAttrOfType<mlir::StringAttr>("kind");
  if (!kindAttr)
    return std::nullopt;
  llvm::StringRef kind = kindAttr.getValue();

  auto readI64 = [&](llvm::StringRef name) -> int64_t {
    return op->getAttrOfType<mlir::IntegerAttr>(name).getInt();
  };
  auto tryReadI64 = [&](llvm::StringRef name) -> std::optional<int64_t> {
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
      return attr.getInt();
    return std::nullopt;
  };

  FlatBlockDotDescriptor d;
  // Group-A block-format geometry (mirrors the typed I4 op attrs).
  d.qk = readI64("qk");
  d.weightStride = readI64("weight_block_stride");
  d.activationStride = readI64("activation_block_stride");
  // Quant offsets: most formats carry `quant_byte_offset` (the weight offset; the
  // activation reuses it unless it has its own `activation_quant_byte_offset`, as
  // the q5 formats do -- their q8 quants sit past a qh/second-scale field). mxfp4
  // instead carries DISTINCT `weight_quant_byte_offset` + `activation_quant_byte_
  // offset` (the E8M0 exponent byte shifts the weight quants off the q8 offset).
  if (auto wq = tryReadI64("weight_quant_byte_offset")) {
    d.quantOffset = *wq;
    d.activationQuantOffset =
        tryReadI64("activation_quant_byte_offset").value_or(*wq);
  } else {
    d.quantOffset = readI64("quant_byte_offset");
    d.activationQuantOffset =
        tryReadI64("activation_quant_byte_offset").value_or(d.quantOffset);
  }
  if (auto high = tryReadI64("activation_high_byte_offset"))
    d.highOffset = *high;
  if (auto qh = tryReadI64("weight_qh_byte_offset")) {
    d.hasQh = true;
    d.qhOffset = *qh;
  }
  if (auto wmin = tryReadI64("weight_min_byte_offset")) {
    d.hasMinTerm = true;
    d.weightMinOffset = *wmin;
  }
  if (auto asum = tryReadI64("activation_sum_byte_offset"))
    d.activationSumOffset = *asum;
  // The 16-entry codebook (DenseI8ArrayAttr) is a STRUCTURAL fact off the op, the
  // 2nd primitive class's table source. Present only on the codebook kinds.
  if (auto cb = op->getAttrOfType<mlir::DenseI8ArrayAttr>("codebook")) {
    d.hasCodebook = true;
    d.codebook = cb.asArrayRef();
  }

  // Group-B primitive + fold + core-LMUL floor, selected by `kind` (unique per
  // format; `scale_model` is NOT sufficient -- q8_0/q4_0/q5_0 share
  // "dual-fp16-per-block-d_x.d_y" but have three distinct fp32 fold trees).
  if (kind == "ggml_q8_0_q8_0_block_dot") {
    d.decodePrimitive = FlatDecodePrimitive::PlainI8;
    d.foldModel = FlatFoldModel::SumiTimesScales;
    d.defaultCoreLmul = "m2";
    d.blockLen = d.qk; // whole 32-element block (no nibble half-split)
  } else if (kind == "ggml_q4_0_q8_0_block_dot") {
    d.decodePrimitive = FlatDecodePrimitive::OffsetBinaryNibble;
    d.foldModel = FlatFoldModel::LeftAssoc;
    d.defaultCoreLmul = "m1";
    d.blockLen = d.qk / 2; // 16 nibble bytes / q8 half lanes per block
  } else if (kind == "ggml_q4_1_q8_1_block_dot") {
    d.decodePrimitive = FlatDecodePrimitive::UnsignedNibble;
    d.foldModel = FlatFoldModel::ScalePlusMin;
    d.defaultCoreLmul = "m1";
    d.blockLen = d.qk / 2;
  } else if (kind == "ggml_q5_0_q8_0_block_dot") {
    d.decodePrimitive = FlatDecodePrimitive::FiveBitOffsetBinary;
    d.foldModel = FlatFoldModel::ScalesTimesSumi;
    d.defaultCoreLmul = "m1";
    d.blockLen = d.qk / 2;
    d.applyOffsetBias = true; // the `-16` offset-binary bias
  } else if (kind == "ggml_q5_1_q8_1_block_dot") {
    d.decodePrimitive = FlatDecodePrimitive::FiveBitOffsetBinary;
    d.foldModel = FlatFoldModel::ScalePlusMin;
    d.defaultCoreLmul = "m1";
    d.blockLen = d.qk / 2;
    d.applyOffsetBias = false; // the bias lives in the per-block MIN scale
  } else if (kind == "ggml_iq4_nl_q8_0_block_dot") {
    // The 2nd primitive class: 16-entry int8 codebook gather. m1 anchor floor
    // (the VLEN-capability gather anchor); the fp32 fold is ggml's iq4_nl
    // scales-first order (SumiTimesScales, the q8_0 tree). The weight scale is a
    // plain fp16 read (same as q8_0/q4_0); the codebook table is broadcast once.
    d.decodePrimitive = FlatDecodePrimitive::CodebookGatherNibble;
    d.foldModel = FlatFoldModel::SumiTimesScales;
    d.defaultCoreLmul = "m1";
    d.blockLen = d.qk / 2; // 16 nibble bytes / q8 half lanes per block
    d.weightScaleSource = FlatWeightScaleSource::Fp16;
    d.codebookTableName = "tcrv_iq4_nl_kvalues";
  } else if (kind == "ggml_mxfp4_q8_0_block_dot") {
    // The FP4-class codebook: the SAME 16-entry gather as iq4_nl, but the weight
    // scale is the structured E8M0 -> fp32 half reconstruction (no fp16 weight
    // field; dual weight/activation quant offsets read above). The fold is ggml's
    // mxfp4 scales-first order (SumiTimesScales, node-identical to iq4_nl's).
    d.decodePrimitive = FlatDecodePrimitive::CodebookGatherNibble;
    d.foldModel = FlatFoldModel::SumiTimesScales;
    d.defaultCoreLmul = "m1";
    d.blockLen = d.qk / 2;
    d.weightScaleSource = FlatWeightScaleSource::E8M0;
    d.codebookTableName = "tcrv_mxfp4_kvalues";
  } else {
    return std::nullopt;
  }
  return d;
}

mlir::LogicalResult VariantToEmitCFunc::emitFlatBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value blockDotResult, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, llvm::StringRef opName,
    llvm::StringRef role, const BlockDotFacts &facts,
    const FlatBlockDotDescriptor &descriptor) const {
  mlir::MLIRContext *ctx = rewriter.getContext();

  // The shared per-block emit state (interned types + wideLmul + bases). The
  // sumf accumulator lvalue and the codebook table register are filled in below
  // once their decls are emitted. buildFlatBlockDotEmitState is the SINGLE
  // source of the interned type spellings, so the monolithic path here and the
  // M-FLAT typed_flat_block_dot_loop_body region driver emit BYTE-IDENTICAL
  // per-block cores.
  FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
      rewriter, descriptor, facts, weightBase, activationBase,
      /*sumfVar=*/mlir::Value(), /*codebookValues=*/mlir::Value(), sizeType,
      opName, role);
  int64_t multiBlockFactor = facts.multiBlockFactor;
  int64_t qk = descriptor.qk;

  auto sizeLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // The 16-entry codebook (2nd primitive class) is a STRUCTURAL fact off the
  // op's DenseI8ArrayAttr. Emit it as a `static const int8_t <name>[N]` decl
  // ONCE, before the accumulator -- the task-sanctioned structured const for the
  // gather table (the decl renders the verified attr entries; the register is
  // broadcast-loaded below the block count and reused by every gather).
  if (descriptor.hasCodebook) {
    std::string decl =
        ("static const int8_t " + descriptor.codebookTableName + "[" +
         std::to_string(descriptor.codebook.size()) + "] = {")
            .str();
    for (size_t i = 0; i < descriptor.codebook.size(); ++i) {
      if (i)
        decl += ", ";
      decl += std::to_string(static_cast<int>(descriptor.codebook[i]));
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);
  }

  // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
  rewriter.create<emitc::VerbatimOp>(
      loc, localVariableComment("sumf", opName, role));
  auto sumfVar = rewriter.create<emitc::VariableOp>(
      loc, emitc::LValueType::get(st.floatType),
      emitc::OpaqueAttr::get(ctx, ""));
  rewriter.create<emitc::AssignOp>(
      loc, sumfVar,
      rewriter.create<emitc::LiteralOp>(loc, st.floatType, "0.0f"));
  st.sumfVar = sumfVar.getResult();

  // size_t nb = n / QK;
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  // The codebook table broadcast into a vector register ONCE (reused by every
  // gather): vint8<L>_t values = vle8_v_i8<L>(<name>, N). The table pointer is
  // the structured-const decl above, spelled at the i8 anchor LMUL. Null for the
  // non-codebook primitives (the gather case is the only reader).
  if (descriptor.hasCodebook) {
    std::string tableLoadCallee = riscvIntrinsicName("vle", 8, st.coreLmul, "i8");
    st.codebookValues = emitOpaqueCallBuilt(
        rewriter, loc, st.i8CoreType, tableLoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value tableName = rewriter.create<emitc::LiteralOp>(
              loc, st.i8PtrType, descriptor.codebookTableName.str());
          return {tableName, sizeLit(descriptor.codebook.size())};
        },
        llvm::StringRef("codebook_table_load"));
  }

  if (multiBlockFactor == 1) {
    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the no-unroll form.
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());
    mlir::FailureOr<FlatBlockCore> core = emitFlatBlockCore(
        rewriter, loc, st, blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
    if (mlir::failed(core))
      return mlir::failure();
    emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY, core->mX,
                 core->sY);
  } else {
    // Multi-block unroll: a main loop stepping by factor over nb - nb%factor
    // full groups -- emit ALL factor independent cores FIRST (the latency-
    // overlap lever), THEN the factor folds in strict ascending block order --
    // then a robust single-block scalar tail over the nb % factor remainder.
    mlir::Value factorLit = sizeLit(multiBlockFactor);
    mlir::Value nbRem =
        rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
    mlir::Value nbMain =
        rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                  factorLit,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      llvm::SmallVector<FlatBlockCore> cores;
      for (int64_t k = 0; k < multiBlockFactor; ++k) {
        mlir::FailureOr<FlatBlockCore> core = emitFlatBlockCore(
            rewriter, loc, st, mainLoop.getInductionVar(), k,
            /*forceRobust=*/false);
        if (mlir::failed(core))
          return mlir::failure();
        cores.push_back(*core);
      }
      for (const FlatBlockCore &core : cores)
        emitFlatFold(rewriter, loc, st, core.sumiVar, core.dX, core.dY, core.mX,
                     core.sY);
    }
    auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      mlir::FailureOr<FlatBlockCore> core = emitFlatBlockCore(
          rewriter, loc, st, tailLoop.getInductionVar(), 0,
          /*forceRobust=*/true);
      if (mlir::failed(core))
        return mlir::failure();
      emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                   core->mX, core->sY);
    }
  }

  // *s = sumf;  (structured scalar store through the output pointer)
  auto outPointer =
      llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
  if (!outPointer)
    return rewriter.notifyMatchFailure(blockDotResult.getDefiningOp(),
                                       "block-dot output not a pointer");
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "store_s"));
  mlir::Value outIndex =
      rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
  emitc::SubscriptOp outSubscript =
      rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
  mlir::Value sumfFinal =
      rewriter.create<emitc::LoadOp>(loc, st.floatType, st.sumfVar).getResult();
  rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

  valueMap[blockDotResult] = sumfFinal;
  return mlir::success();
}

// Build the shared per-block emit state both flat-block-dot callers use. The
// interned emitc type spellings + the wideLmul superset formula are single-
// sourced here so the monolithic emitFlatBlockDot and the M-FLAT loop-body
// driver emit BYTE-IDENTICAL per-block cores. weightPtrType/activationPtrType
// come straight off the imported ABI base pointer types; sumfVar/codebookValues
// are filled in by the caller after their decls are emitted.
FlatBlockDotEmitState VariantToEmitCFunc::buildFlatBlockDotEmitState(
    mlir::ConversionPatternRewriter &rewriter,
    const FlatBlockDotDescriptor &descriptor, const BlockDotFacts &facts,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value sumfVar,
    mlir::Value codebookValues, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  FlatBlockDotEmitState st;
  st.descriptor = descriptor;
  st.opName = opName;
  st.role = role;
  st.coreLmul = facts.coreLmul;
  // i8 source LMUL -> the next-wider i16 product LMUL (m2->m4, m1->m2, mf2->m1,
  // mf4->mf2). Byte-exact for every in-tree anchor.
  st.wideLmul =
      (facts.coreLmul == "m2")    ? "m4"
      : (facts.coreLmul == "m1")  ? "m2"
      : (facts.coreLmul == "mf2") ? "m1"
                                  : "mf2";
  st.stripElided = facts.stripElided;
  st.weightBase = weightBase;
  st.activationBase = activationBase;
  st.sumfVar = sumfVar;
  st.codebookValues = codebookValues;
  st.sizeType = sizeType;
  st.floatType = emitc::OpaqueType::get(ctx, "float");
  st.i32Type = emitc::OpaqueType::get(ctx, "int32_t");
  st.u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
  st.weightPtrType = weightBase.getType();
  st.activationPtrType = activationBase.getType();
  std::string i8CoreTypeName = ("vint8" + facts.coreLmul + "_t").str();
  std::string u8CoreTypeName = ("vuint8" + facts.coreLmul + "_t").str();
  std::string i16WideTypeName = ("vint16" + st.wideLmul + "_t").str();
  std::string u16WideTypeName = ("vuint16" + st.wideLmul + "_t").str();
  st.i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
  st.u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
  st.i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
  st.u16WideType = emitc::OpaqueType::get(ctx, u16WideTypeName);
  st.i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
  st.i8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
  st.u8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
  st.fp16ReadCallee = "(float)*(const _Float16 *)";
  st.u16ReadCallee = "(uint16_t)*(const uint16_t *)";
  return st;
}

// Factored VERBATIM from emitFlatBlockDot's former emitIntegerCore + the strip
// reduce it drives; reads all shared state off `st`. Byte-identical emit.
mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitFlatIntegerCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const FlatBlockDotEmitState &st, mlir::Value xb, mlir::Value yb,
    mlir::Value qhLow16, mlir::Value qhHigh16, bool forceRobust) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  const FlatBlockDotDescriptor &descriptor = st.descriptor;
  llvm::StringRef opName = st.opName;
  llvm::StringRef role = st.role;
  llvm::StringRef coreLmul = st.coreLmul;
  llvm::StringRef wideLmul = st.wideLmul;
  bool stripElided = st.stripElided;
  int64_t blockLen = descriptor.blockLen;
  int64_t quantOffset = descriptor.quantOffset;
  int64_t actQuantOffset = descriptor.activationQuantOffset;
  int64_t highOffset = descriptor.highOffset;
  mlir::Type sizeType = st.sizeType;
  mlir::Type i32Type = st.i32Type;
  mlir::Type weightPtrType = st.weightPtrType;
  mlir::Type activationPtrType = st.activationPtrType;
  mlir::Type i8CoreType = st.i8CoreType;
  mlir::Type u8CoreType = st.u8CoreType;
  mlir::Type i16WideType = st.i16WideType;
  mlir::Type u16WideType = st.u16WideType;
  mlir::Type i32m1Type = st.i32m1Type;
  mlir::Type i8PtrType = st.i8PtrType;
  mlir::Type u8PtrType = st.u8PtrType;
  mlir::Value codebookValues = st.codebookValues;
  auto sizeLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
  };

    // The decode+product for ONE strip -- the ONLY arithmetic divergence, a
    // switch over the EXISTING factored decode helpers -- followed by the SHARED
    // seed/vwredsum/extract tail. qhLow16/qhHigh16 are null for the non-five-bit
    // primitives. chunkOffset is the within-block byte offset (the strip
    // induction var for the robust loop, or a 0 literal for the elided core).
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value qhLow16, mlir::Value qhHigh16,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          mlir::Type castPtrType, int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castPtrType, full)
            .getResult();
      };
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      auto loadI8 = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      auto loadU8 = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };

      mlir::FailureOr<mlir::Value> productOr = mlir::failure();
      switch (descriptor.decodePrimitive) {
      case FlatDecodePrimitive::PlainI8: {
        // Plain signed widening product: i8 x i8 -> i16 (NO nibble decode).
        mlir::Value vx0 =
            loadI8(chunkPtr(xb, weightPtrType, i8PtrType, quantOffset));
        mlir::Value vy0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        std::string mulCallee =
            riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
        productOr =
            emitOpaqueCall(rewriter, loc, i16WideType, mulCallee,
                           mlir::ValueRange{vx0, vy0, vl}, opName, role);
        break;
      }
      case FlatDecodePrimitive::OffsetBinaryNibble: {
        // Offset-binary asymmetric i4xi8 (weight loaded signed i8; xor 0x88 +
        // sll/sra sign-extend + vwmul/vwmacc against the low/high q8 halves).
        mlir::Value w =
            loadI8(chunkPtr(xb, weightPtrType, i8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        productOr = emitOffsetBinaryDecodeProductValue(
            rewriter, loc, w, y0, y1, vl, i8CoreType, i16WideType, "i8",
            coreLmul, 16, wideLmul, "i16", opName, role);
        break;
      }
      case FlatDecodePrimitive::UnsignedNibble: {
        // Unsigned-nibble asymmetric i4xi8 (weight loaded u8; vand 0x0F / vsrl
        // 0x04 + reinterpret to i8 + vwmul/vwmacc).
        mlir::Value w =
            loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        productOr = emitUnsignedNibbleDecodeProductValue(
            rewriter, loc, w, y0, y1, vl, i8CoreType, u8CoreType, i16WideType,
            coreLmul, 16, wideLmul, "i16", opName, role);
        break;
      }
      case FlatDecodePrimitive::FiveBitOffsetBinary: {
        // 5-bit offset-binary (weight loaded u8 at quantOffset; q8 halves at the
        // DISTINCT actQuantOffset; nibble + qh 5th-bit merge; applyOffsetBias
        // selects the `-16` for q5_0 vs the MIN-scale bias for q5_1).
        mlir::Value w =
            loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        productOr = emitFiveBitOffsetBinaryDecodeProductValue(
            rewriter, loc, w, y0, y1, qhLow16, qhHigh16, chunkOffset, vl,
            i8CoreType, u8CoreType, u16WideType, i16WideType, coreLmul, wideLmul,
            16, wideLmul, "i16", opName, role, descriptor.applyOffsetBias);
        break;
      }
      case FlatDecodePrimitive::CodebookGatherNibble: {
        // 2nd primitive class: split the packed weight byte into the two UNSIGNED
        // nibble index lanes (vand 0x0F / vsrl 0x04), GATHER each through the
        // broadcast codebook table (vrgather_vv_i8<L>) into signed-i8 weight lanes
        // v0/v1, then feed the SAME asymmetric signed widening product the
        // offset-binary sibling uses (vwmul low <-> q8[0..15], vwmacc + high).
        mlir::Value w =
            loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                           llvm::StringRef amount) -> mlir::Value {
          std::string callee = ("__riscv_" + mnemonic + "_u8" + coreLmul).str();
          return emitOpaqueCallBuilt(
              rewriter, loc, u8CoreType, callee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value amt = rewriter.create<emitc::LiteralOp>(
                    loc, emitc::OpaqueType::get(ctx, "int"), amount.str());
                return {src, amt, vl};
              });
        };
        mlir::Value idxLow = u8ImmOp("vand_vx", w, "0x0F");
        mlir::Value idxHigh = u8ImmOp("vsrl_vx", w, "0x04");
        std::string gatherCallee = ("__riscv_vrgather_vv_i8" + coreLmul).str();
        auto gather = [&](mlir::Value idx) -> mlir::Value {
          return emitOpaqueCall(rewriter, loc, i8CoreType, gatherCallee,
                                mlir::ValueRange{codebookValues, idx, vl}, opName,
                                role);
        };
        mlir::Value v0 = gather(idxLow);
        mlir::Value v1 = gather(idxHigh);
        productOr = emitOffsetBinaryProductFromDecodedValue(
            rewriter, loc, v0, v1, y0, y1, vl, i16WideType, 16, wideLmul, "i16",
            opName, role);
        break;
      }
      }
      if (mlir::failed(productOr))
        return mlir::failure();
      mlir::Value product = *productOr;

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, seedCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumiSeed =
                carrySumi
                    ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
            mlir::Value one = sizeLit(1);
            return {sumiSeed, one};
          });
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red =
          emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                         mlir::ValueRange{product, seed, vl}, opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
    };
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The vsetvl SEW/LMUL spelling of the anchor: m1/m2 use vsetvl_e8<lmul>
      // (byte strips), mf4 uses vsetvl_e32m1 (4-element strips at VLEN=128).
      unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
      llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");

      if (!forceRobust && stripElided) {
        // Elided core (VLEN >= guaranteed floor): ONE vsetvl(block_len) caps the
        // active vl at the whole (half-)block + ONE strip reduce. NO inner strip
        // loop, NO sumi carry (seed lane0 = 0).
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, qhLow16, qhHigh16, sizeLit(0), vl, sumiVar,
                            /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the block_len bytes; the loop STEP
      // is the loop-invariant VLMAX, the per-chunk active vl is vsetvl(len - c).
      // Stays VLEN-robust via the sumi-carrying seed.
      mlir::Value innerVlmax = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(blockLen)};
          });
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining = rewriter.create<emitc::SubOp>(
                  loc, sizeType, sizeLit(blockLen), c);
              return {remaining};
            });

        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, qhLow16, qhHigh16, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
}

// Factored VERBATIM from emitFlatBlockDot's former emitFold; reads shared state
// off `st`. The fold tree switches on st.descriptor.foldModel and folds the
// dX/dY (+ mX/sY) OPERANDS -- region-driven, not attribute-rederived.
void VariantToEmitCFunc::emitFlatFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const FlatBlockDotEmitState &st, mlir::Value sumiVar, mlir::Value dX,
    mlir::Value dY, mlir::Value mX, mlir::Value sY) const {
  const FlatBlockDotDescriptor &descriptor = st.descriptor;
  llvm::StringRef opName = st.opName;
  llvm::StringRef role = st.role;
  mlir::Type floatType = st.floatType;
  mlir::Type i32Type = st.i32Type;
  mlir::Value sumfVar = st.sumfVar;

      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();

      // Pinned fp-fold oracle [K-5]/[testing/flat-block-dot-fp-fold-oracle.md §1]:
      // strict left-assoc, ordered, NO dx*dy premultiply, NO FMA contraction.
      // Emitted as SEPARATE emitc statements (standalone cast/mul/mul/add, NOT
      // inside one emitc.expression) so clang's default -ffp-contract=on cannot
      // fuse the (t*d_y)+sumf into an fmaf -- cross-statement contraction with
      // named intermediates is not permitted. The two muls give ((sumi*d_x)*d_y),
      // the add folds into the ordered running sumf:
      //   float t = (float)sumi * d_x;   // t = f32(sumi) (x) dx
      //   t = t * d_y;                   // t = t (x) dy
      //   sumf = sumf + t;               // ordered (+)
      if (descriptor.foldModel == FlatFoldModel::SeparatedLeftAssoc) {
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value t =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
        mlir::Value t2 = rewriter.create<emitc::MulOp>(loc, floatType, t, dY);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, t2);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumf", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfNext);
        return;
      }

      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumfNext;
        switch (descriptor.foldModel) {
        case FlatFoldModel::SumiTimesScales: {
          // sumf + (float)sumi * (d_x * d_y)  (ggml q8_0 order: scales FIRST).
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value scaleProduct =
              rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
          mlir::Value blockTerm = rewriter.create<emitc::MulOp>(
              loc, floatType, sumiFloat, scaleProduct);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::LeftAssoc: {
          // sumf + ((float)sumi * d_x) * d_y  (ggml q4_0 left-assoc order).
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value timesDx =
              rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
          mlir::Value blockTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::ScalesTimesSumi: {
          // sumf + (d_x * d_y) * (float)sumi  (ggml q5_0 order: scales FIRST as
          // the LEFT operand -- distinct emitc sequence from q8_0's above).
          mlir::Value scales =
              rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value blockTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, scales, sumiFloat);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::ScalePlusMin: {
          // sumf + ((d_x*d_y)*sumi + m_x*s_y)  (ggml q4_1/q5_1 exact tree: the
          // two products are SUMMED FIRST, then added to sumf).
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value scaleProduct =
              rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
          mlir::Value scaleTerm = rewriter.create<emitc::MulOp>(
              loc, floatType, scaleProduct, sumiFloat);
          mlir::Value minTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, mX, sY);
          mlir::Value blockTerm =
              rewriter.create<emitc::AddOp>(loc, floatType, scaleTerm, minTerm);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::SeparatedLeftAssoc:
          // Handled above via SEPARATE emitc statements (no fused expression),
          // so it never reaches this expression-body switch.
          llvm_unreachable(
              "SeparatedLeftAssoc is emitted before the fused-expression switch");
        }
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
}

// Factored VERBATIM from emitFlatBlockDot's former emitBlockCore + its per-block
// address / fp16-read / qh helpers; reads shared state off `st`. Byte-identical
// emit. The integer core is driven through the SHARED emitFlatIntegerCore.
mlir::FailureOr<FlatBlockCore> VariantToEmitCFunc::emitFlatBlockCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const FlatBlockDotEmitState &st, mlir::Value ib, int64_t blockOffset,
    bool forceRobust) const {
  const FlatBlockDotDescriptor &descriptor = st.descriptor;
  llvm::StringRef opName = st.opName;
  llvm::StringRef role = st.role;
  mlir::Type sizeType = st.sizeType;
  mlir::Type floatType = st.floatType;
  mlir::Type u32Type = st.u32Type;
  mlir::Type weightPtrType = st.weightPtrType;
  mlir::Type activationPtrType = st.activationPtrType;
  mlir::Value weightBase = st.weightBase;
  mlir::Value activationBase = st.activationBase;
  int64_t weightStride = descriptor.weightStride;
  int64_t activationStride = descriptor.activationStride;
  llvm::StringRef fp16ReadCallee = st.fp16ReadCallee;
  llvm::StringRef u16ReadCallee = st.u16ReadCallee;
  auto sizeLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
  };

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*Sw;
    // const uint8_t *yb = vy + (ib+blockOffset)*Sa.
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };
    // A scalar fp16->fp32 read at a per-block byte offset (the sanctioned opaque
    // piece). At byteOffset 0 no emitc.add is emitted, so this is byte-identical
    // to the family-A `fp16Read(base)` form and to the family-B `fp16ReadAt`.
    auto fp16ReadAt = [&](mlir::Value blockBase, mlir::Type ptrType,
                          int64_t byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, ptrType, blockBase,
                                             sizeLit(byteOffset));
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{addr}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };
    // The per-element 5th-bit qh field (five-bit primitive only), read as two
    // ALIGNED 16-bit halves (LE (qh & 0xFFFF) / (qh >> 16)).
    struct QhHalves {
      mlir::Value low16;
      mlir::Value high16;
    };
    auto qhRead = [&](mlir::Value xb) -> QhHalves {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qh_field"));
      auto readHalf = [&](int64_t byteOffset) -> mlir::Value {
        mlir::Value ptr = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, xb, sizeLit(byteOffset));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                         u16ReadCallee, mlir::ValueRange{ptr})
            .getResult(0);
      };
      mlir::Value low16 = readHalf(descriptor.qhOffset);
      mlir::Value high16 = readHalf(descriptor.qhOffset + 2);
      return QhHalves{low16, high16};
    };
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      // The weight scale: the sanctioned fp16 read (every flat-plain + iq4_nl
      // format) or the structured E8M0 -> fp32 half reconstruction (mxfp4, no
      // fp16 weight field). The activation scale is always the fp16 read.
      mlir::Value dX =
          (descriptor.weightScaleSource == FlatWeightScaleSource::E8M0)
              ? emitE8M0HalfScale(rewriter, loc, xb, opName, role)
              : fp16ReadAt(xb, weightPtrType, 0);
      mlir::Value dY = fp16ReadAt(yb, activationPtrType, 0);
      mlir::Value mX = nullptr;
      mlir::Value sY = nullptr;
      if (descriptor.hasMinTerm) {
        mX = fp16ReadAt(xb, weightPtrType, descriptor.weightMinOffset);
        sY = fp16ReadAt(yb, activationPtrType, descriptor.activationSumOffset);
      }
      mlir::Value qhLow16 = nullptr;
      mlir::Value qhHigh16 = nullptr;
      if (descriptor.hasQh) {
        QhHalves qh = qhRead(xb);
        qhLow16 = qh.low16;
        qhHigh16 = qh.high16;
      }
      mlir::FailureOr<mlir::Value> sumiVar =
          emitFlatIntegerCore(rewriter, loc, st, xb, yb, qhLow16, qhHigh16, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return FlatBlockCore{*sumiVar, dX, dY, mX, sY};
}


// M-FLAT loop-scaffold step 1/6: lower the region-carrying
// tcrv_rvv.typed_flat_block_dot_loop_body to the byte-exact SKELETON that
// emitFlatBlockDot emits for its mbf==1 form. The isolated hard bone this step
// proves is the SSA loop-carried f32 accumulator -> emitc mutable-variable
// mapping: emitc.for has no iter_args, so the region's carried-IN `acc` block
// argument maps to a LOAD of the sumf lvalue at the top of the loop body and
// the typed loop-yield's carried-OUT `acc_next` maps to an emitc.assign back
// into it at the bottom (the SCFToEmitC / F3-F6 loop-carried-scalar pattern).
// The seed is the literal `0.0f` emitted directly (ggml's `float sumf = 0.0f;`
// is a hardcoded zero, not a caller value; the op carries no init operand,
// mirroring the monolithic block-dot ops). The minimal region body is a single
// tcrv_rvv.cross_block_f32_accumulate (brick 3) over a stub term, dispatched
// through the existing brick emitter; the full per-block primitive chain and
// full-body byte-exactness are later steps. Only the mbf==1 skeleton form is
// lowered here; any unroll form is fail-closed (I7).
mlir::LogicalResult VariantToEmitCFunc::emitTypedFlatBlockDotLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");

  tcrvrvv::TypedFlatBlockDotLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<tcrvrvv::TypedFlatBlockDotLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed flat block-dot loop body missing the op");

  // M-FLAT schedule-parametrization step 1: the loop-body's bounded scheduling
  // knobs (I7-bounded by the op verifier: multi_block_factor in {1,2,4},
  // strip_elision in {robust,elided}). These drive the *how* (outer-loop unroll /
  // inner strip form); the region bricks drive the *what*. The q4_0 (left_assoc)
  // flat body below materializes the FULL {integer_core_lmul, multi_block_factor,
  // strip_elision} cross product byte-exact to the monolithic emitFlatBlockDot;
  // the other folds still require the mbf==1 + elided default (fail-closed).
  int64_t multiBlockFactor = loopBody.getMultiBlockFactor().value_or(1);
  bool stripElided = loopBody.getStripElision().value_or("robust") == "elided";

  mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
  mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
  mlir::Value output = valueMap.lookup(loopBody.getOutput());
  if (!weightBase || !activationBase || !output)
    return rewriter.notifyMatchFailure(loopBody,
                                       "loop-body ABI operand unmapped");

  llvm::StringRef opName = loopBody.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getTCRVEmitCLowerableSourceRole();
  int64_t qk = loopBody.getQk();

  auto sizeLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // float sumf = 0.0f;  -- byte-exact to emitFlatBlockDot's sumf accumulator
  // decl (:5410-5417). The SSA loop-carried acc lowers to this mutable
  // emitc.variable lvalue.
  rewriter.create<emitc::VerbatimOp>(
      loc, localVariableComment("sumf", opName, role));
  auto sumfVar = rewriter.create<emitc::VariableOp>(
      loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
  rewriter.create<emitc::AssignOp>(
      loc, sumfVar,
      rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

  // size_t nb = n / QK;  -- byte-exact to emitFlatBlockDot:5419-5422. n is the
  // scope AVL, exactly as the monolithic body derives it.
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  // Peek the region format WITHOUT emitting: a full body carries brick 2
  // (the computed-scale dequant); the q4_0 (left_assoc) full body is the one the
  // schedule-parametrization step materializes across ALL legal knob combos.
  tcrvrvv::BlockComputedScaleDequantOp peekBrick2;
  loopBody.getBody().walk(
      [&](tcrvrvv::BlockComputedScaleDequantOp o) { peekBrick2 = o; });
  const bool isQ40ScheduleParam =
      peekBrick2 && loopBody.getFoldModel() == "left_assoc";
  // The q8_0 (sumi_times_scales) full body is the SECOND flat fold whose FULL
  // legal {integer_core_lmul, multi_block_factor, strip_elision} cross product is
  // schedule-parameterized (ported from the q4_0 scaffold). q8_0's native anchor
  // is the whole-block plain-i8 m2 core (blockLen = qk, two i8 loads, direct
  // vwmul), so its cross product is {m2}×mbf{1,2,4}×strip{robust,elided}.
  const bool isQ80ScheduleParam =
      peekBrick2 && loopBody.getFoldModel() == "sumi_times_scales";

  if (isQ40ScheduleParam) {
    // ===================================================================
    // q4_0 (left_assoc) SCHEDULE-PARAMETERIZED emit: the FULL legal
    // {integer_core_lmul, multi_block_factor, strip_elision} cross product,
    // driven by the loop-body knobs but sourced OP-BY-OP from the region ops
    // (the brick chain drives WHAT computes; the knobs drive HOW it schedules).
    // The schedule scaffold mirrors the monolithic emitFlatBlockDot (mbf-unroll
    // main loop + strict-ascending folds + robust tail; the inner strip loop for
    // the robust form), so each legal combo is byte-identical to the monolith's
    // same-knob GgmlBlockDotQ40Op instance while the integer core stays
    // region-driven (anti-bypass W4). ===================================
    mlir::Block &coreBlock = loopBody.getBody().front();

    // ---- Region walk (identify, no emit) ----
    tcrvrvv::BlockFp16ScaleProductOp brick1;
    tcrvrvv::BlockComputedScaleDequantOp brick2 = peekBrick2;
    tcrvrvv::CrossBlockF32AccumulateOp brick3;
    tcrvrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    tcrvrvv::StandaloneReduceOp coreReduce;
    tcrvrvv::TypedVectorLane0ToScalarExtractOp coreExtract;
    tcrvrvv::PackedI4OffsetBinaryXI8ProductOp packedProduct;
    llvm::SmallVector<tcrvrvv::LoadOp, 3> coreLoads;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<tcrvrvv::BlockFp16ScaleProductOp>(bodyOp))
        brick1 = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::CrossBlockF32AccumulateOp>(bodyOp))
        brick3 = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(bodyOp))
        coreReduce = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedVectorLane0ToScalarExtractOp>(
                       bodyOp))
        coreExtract = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::PackedI4OffsetBinaryXI8ProductOp>(
                       bodyOp))
        packedProduct = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::LoadOp>(bodyOp))
        coreLoads.push_back(o);
    });

    // ---- Region-driven gate (fail-closed, I7): the same brick-wiring checks the
    // mbf==1 path runs, so the schedule-parameterized emit provably tracks the
    // region content, not the attrs. ----
    if (!brick1 || !brick3 || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q4_0 schedule-parameterized body requires brick 1, brick 2, "
                    "brick 3, and the yield");
    if (brick1.getBlockIndex() != coreBlock.getArgument(0))
      return rewriter.notifyMatchFailure(
          brick1, "brick 1 block_index must be the loop induction variable");
    if (brick2.getComputedScale() != brick1.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 computed_scale must be brick 1's per-block scale");
    if (brick3.getAcc() != coreBlock.getArgument(1))
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 acc must be the loop-carried accumulator");
    if (brick3.getTerm() != brick2.getResult())
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 term must be brick 2's per-block dequant term");
    if (yieldOp.getAccNext() != brick3.getResult())
      return rewriter.notifyMatchFailure(
          yieldOp, "loop yield acc_next must be brick 3's fold result");
    if (!packedProduct || coreLoads.size() != 3 || !coreExtract)
      return rewriter.notifyMatchFailure(
          loopBody,
          "full q4_0 flat block-dot body requires the region integer core: three "
          "per-block i8 loads (a packed-i4 weight + two plain-i8 q8 halves), an "
          "asymmetric offset-binary packed-i4 x i8 product, and a lane0 scalar "
          "extract");
    if (brick2.getSumi() != coreExtract.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 sumi must be the integer-core lane0 extract result");
    auto q40WeightLoad = packedProduct.getWeight().getDefiningOp<tcrvrvv::LoadOp>();
    auto q40LowLoad =
        packedProduct.getActivationLow().getDefiningOp<tcrvrvv::LoadOp>();
    auto q40HighLoad =
        packedProduct.getActivationHigh().getDefiningOp<tcrvrvv::LoadOp>();
    if (!q40WeightLoad || !q40LowLoad || !q40HighLoad ||
        q40WeightLoad.getBuffer() != loopBody.getWeightBase() ||
        q40LowLoad.getBuffer() != loopBody.getActivationBase() ||
        q40HighLoad.getBuffer() != loopBody.getActivationBase() ||
        !q40WeightLoad.getQuantByteOffset() || !q40LowLoad.getQuantByteOffset() ||
        !q40HighLoad.getQuantByteOffset() || !q40WeightLoad.getBlockStride() ||
        !q40LowLoad.getBlockStride())
      return rewriter.notifyMatchFailure(
          loopBody, "q4_0 packed-i4 product operands must be the region's "
                    "per-block loads off the ABI buffers with block_stride + "
                    "quant_byte_offset");
    if (!coreReduce || coreReduce.getInput() != packedProduct.getResult())
      return rewriter.notifyMatchFailure(
          coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
          "q4_0 integer-core reduce input must be the packed-i4 product");
    if (coreExtract.getInput() != coreReduce.getResult())
      return rewriter.notifyMatchFailure(
          coreExtract, "q4_0 integer-core lane0 extract input must be the reduce");

    // ---- The LeftAssoc fold descriptor + shared emit state (fold only; the
    // integer core is emitted op-by-op below). ----
    FlatBlockDotDescriptor descriptor;
    descriptor.decodePrimitive = FlatDecodePrimitive::OffsetBinaryNibble;
    descriptor.foldModel = FlatFoldModel::LeftAssoc;
    descriptor.defaultCoreLmul = "m1";
    BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m1");
    FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
        rewriter, descriptor, facts, weightBase, activationBase,
        sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType, opName,
        role);

    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    llvm::StringRef coreLmul =
        llvm::cast<tcrvrvv::VectorType>(q40WeightLoad.getLoaded().getType())
            .getLmul();
    llvm::StringRef wideLmul =
        llvm::cast<tcrvrvv::VectorType>(packedProduct.getResult().getType())
            .getLmul();
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
    llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
    std::string innerSetvlCallee = riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
    int64_t blockLen = qk / 2; // q4_0 half-block strip.

    // ---- The per-block q4_0 core, parameterized by (ib, blockOffset,
    // forceRobust). Emits address / brick-1 scales / the op-by-op integer core
    // (elided single cover OR robust inner strip loop) and returns the fold
    // inputs; the FOLD is emitted by the scaffold (so an mbf-unroll group emits
    // all cores first, then the folds in strict ascending order). ----
    auto emitQ40Core =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<FlatBlockCore> {
      // Fresh per-core base memo: each unrolled core (blockOffset k) recomputes
      // base + (ib+k)*stride; a memo shared across cores would alias block k>0 to
      // block 0's address (byte-diff AND a correctness bug).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value idx = ib;
        if (blockOffset != 0)
          idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                              sizeLit(blockOffset));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };
      mlir::Value xb = blockBaseFor(
          q40WeightLoad.getBuffer(), q40WeightLoad.getBlockIndex(),
          static_cast<int64_t>(*q40WeightLoad.getBlockStride()), "block_base_x");
      mlir::Value yb = blockBaseFor(
          q40LowLoad.getBuffer(), q40LowLoad.getBlockIndex(),
          static_cast<int64_t>(*q40LowLoad.getBlockStride()), "block_base_y");
      auto fp16ReadAt = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
        mlir::Value addr = blockBase;
        if (byteOffset && *byteOffset != 0)
          addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                               blockBase, sizeLit(*byteOffset));
        return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                              mlir::ValueRange{addr}, opName, role,
                              llvm::StringRef("fcvt.s.h"));
      };
      mlir::Value dX = fp16ReadAt(
          blockBaseFor(brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                       "block_base_x"),
          brick1.getLhsScaleByteOffset());
      mlir::Value dY = fp16ReadAt(
          blockBaseFor(brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                       "block_base_y"),
          brick1.getRhsScaleByteOffset());

      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // One strip: op-sourced loads + offset-binary decode product + reduce +
      // lane0 extract + assign sumi. chunkOffset is the elided 0 literal or the
      // robust strip induction var; carrySumi seeds lane0 from the sumi lvalue
      // (robust re-strip) or a fresh 0 (elided single cover).
      auto emitStrip = [&](mlir::Value chunkOffset, mlir::Value vl,
                           bool carrySumi) -> mlir::LogicalResult {
        auto emitLoadTail = [&](mlir::Value blockBase,
                                int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[q40WeightLoad.getLoaded()] = emitLoadTail(
            xb, static_cast<int64_t>(*q40WeightLoad.getQuantByteOffset()));
        valueMap[q40LowLoad.getLoaded()] = emitLoadTail(
            yb, static_cast<int64_t>(*q40LowLoad.getQuantByteOffset()));
        valueMap[q40HighLoad.getLoaded()] = emitLoadTail(
            yb, static_cast<int64_t>(*q40HighLoad.getQuantByteOffset()));
        mlir::FailureOr<mlir::Value> productOr =
            emitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(packedProduct.getWeight()),
                valueMap.lookup(packedProduct.getActivationLow()),
                valueMap.lookup(packedProduct.getActivationHigh()), vl,
                i8CoreType, i16WideType, "i8", coreLmul, 16, wideLmul, "i16",
                opName, role);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[packedProduct.getResult()] = *productOr;
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value sumiSeed =
                  carrySumi
                      ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                            .getResult()
                      : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                            .getResult();
              return {sumiSeed, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, "__riscv_vmv_x_s_i32m1_i32",
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);
        return mlir::success();
      };

      bool robust = forceRobust || !stripElided;
      if (!robust) {
        // Elided single cover: ONE vsetvl(block_len) caps the active vl at the
        // whole half-block, ONE strip reduce, no inner loop, no sumi carry.
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        if (mlir::failed(emitStrip(sizeLit(0), vl, /*carrySumi=*/false)))
          return mlir::failure();
      } else {
        // Robust inner strip loop: step = loop-invariant VLMAX, per-chunk active
        // vl = vsetvl(block_len - c), sumi-carrying seed. VLEN-robust re-strip.
        mlir::Value innerVlmax = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        auto innerLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
            /*bodyBuilder=*/nullptr);
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining = rewriter.create<emitc::SubOp>(
                  loc, sizeType, sizeLit(blockLen), c);
              return {remaining};
            });
        if (mlir::failed(emitStrip(c, vl, /*carrySumi=*/true)))
          return mlir::failure();
      }
      return FlatBlockCore{sumiVar.getResult(), dX, dY, mlir::Value(),
                           mlir::Value()};
    };

    // ---- The mbf schedule scaffold (mirrors emitFlatBlockDot): mbf==1 = single
    // block loop {core; fold}; mbf>1 = a main loop stepping by factor emitting
    // ALL factor cores FIRST then the factor folds in STRICT ascending block
    // order (the fp non-associativity boundary), plus a robust single-block
    // scalar tail over the nb % factor remainder. ----
    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<FlatBlockCore> core =
          emitQ40Core(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                   core->mX, core->sY);
    } else {
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nbMain, factorLit, /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<FlatBlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<FlatBlockCore> core =
              emitQ40Core(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const FlatBlockCore &core : cores)
          emitFlatFold(rewriter, loc, st, core.sumiVar, core.dX, core.dY,
                       core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(
          loc, nbMain, nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<FlatBlockCore> core =
            emitQ40Core(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                     core->mX, core->sY);
      }
    }
  } else if (isQ80ScheduleParam) {
    // ===================================================================
    // q8_0 (sumi_times_scales) SCHEDULE-PARAMETERIZED emit: the FULL legal
    // {integer_core_lmul=m2, multi_block_factor, strip_elision} cross product,
    // driven by the loop-body knobs but sourced OP-BY-OP from the region ops
    // (the brick chain drives WHAT computes; the knobs drive HOW it schedules).
    // Ported from the q4_0 schedule scaffold: q8_0's core is the whole-block
    // plain-i8 signed widening product (m2 native, blockLen = qk, two i8 loads,
    // direct vwmul) instead of the half-block packed-i4 decode; the mbf-unroll
    // main loop + strict-ascending folds + robust tail scaffold is structurally
    // identical, so each legal combo is byte-identical to the monolith's
    // same-knob GgmlBlockDotQ80Q80Op (emitFlatBlockDot) instance while the
    // integer core stays region-driven (anti-bypass W4). ==================
    mlir::Block &coreBlock = loopBody.getBody().front();

    // ---- Region walk (identify, no emit) ----
    tcrvrvv::BlockFp16ScaleProductOp brick1;
    tcrvrvv::BlockComputedScaleDequantOp brick2 = peekBrick2;
    tcrvrvv::CrossBlockF32AccumulateOp brick3;
    tcrvrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    tcrvrvv::WideningProductOp coreProduct;
    tcrvrvv::StandaloneReduceOp coreReduce;
    tcrvrvv::TypedVectorLane0ToScalarExtractOp coreExtract;
    llvm::SmallVector<tcrvrvv::LoadOp, 2> coreLoads;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<tcrvrvv::BlockFp16ScaleProductOp>(bodyOp))
        brick1 = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::CrossBlockF32AccumulateOp>(bodyOp))
        brick3 = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::WideningProductOp>(bodyOp))
        coreProduct = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(bodyOp))
        coreReduce = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedVectorLane0ToScalarExtractOp>(
                       bodyOp))
        coreExtract = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::LoadOp>(bodyOp))
        coreLoads.push_back(o);
    });

    // ---- Region-driven gate (fail-closed, I7): the SAME brick + integer-core
    // wiring the mbf==1 else path checks, so the schedule-parameterized emit
    // provably tracks the region content, not the attrs. ----
    if (!brick1 || !brick3 || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q8_0 schedule-parameterized body requires brick 1, brick 2, "
                    "brick 3, and the yield");
    if (brick1.getBlockIndex() != coreBlock.getArgument(0))
      return rewriter.notifyMatchFailure(
          brick1, "brick 1 block_index must be the loop induction variable");
    if (brick2.getComputedScale() != brick1.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 computed_scale must be brick 1's per-block scale");
    if (brick3.getAcc() != coreBlock.getArgument(1))
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 acc must be the loop-carried accumulator");
    if (brick3.getTerm() != brick2.getResult())
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 term must be brick 2's per-block dequant term");
    if (yieldOp.getAccNext() != brick3.getResult())
      return rewriter.notifyMatchFailure(
          yieldOp, "loop yield acc_next must be brick 3's fold result");
    if (coreLoads.size() != 2 || !coreProduct || !coreReduce || !coreExtract)
      return rewriter.notifyMatchFailure(
          loopBody,
          "full q8_0 flat block-dot body requires the region integer core: two "
          "per-block i8 loads, a signed widening product, a standalone reduce, "
          "and a lane0 scalar extract");
    if (brick2.getSumi() != coreExtract.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 sumi must be the integer-core lane0 extract result");

    // Identify the weight vs activation per-block load by ABI buffer; both must
    // carry the per-block block_stride + quant_byte_offset the address
    // arithmetic depends on (the operand-flow real gate).
    tcrvrvv::LoadOp weightLoad, activationLoad;
    for (tcrvrvv::LoadOp ld : coreLoads) {
      if (ld.getBuffer() == loopBody.getWeightBase())
        weightLoad = ld;
      else if (ld.getBuffer() == loopBody.getActivationBase())
        activationLoad = ld;
    }
    if (!weightLoad || !activationLoad || !weightLoad.getQuantByteOffset() ||
        !activationLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
        !activationLoad.getBlockStride())
      return rewriter.notifyMatchFailure(
          loopBody, "q8_0 integer-core loads must read the weight and activation "
                    "ABI buffers off the loop body, each carrying a block_stride "
                    "+ quant_byte_offset");
    if (coreProduct.getKind() != "signed_widening_product")
      return rewriter.notifyMatchFailure(
          coreProduct, "q8_0 integer-core widening product must be a signed "
                       "widening product");
    mlir::Value pl = coreProduct.getLhs(), pr = coreProduct.getRhs();
    mlir::Value wLoaded = weightLoad.getLoaded(),
                aLoaded = activationLoad.getLoaded();
    if (!((pl == wLoaded && pr == aLoaded) ||
          (pl == aLoaded && pr == wLoaded)))
      return rewriter.notifyMatchFailure(
          coreProduct, "q8_0 integer-core widening product operands must be the "
                       "two per-block load results");
    if (coreReduce.getInput() != coreProduct.getResult())
      return rewriter.notifyMatchFailure(
          coreReduce,
          "q8_0 integer-core reduce input must be the widening product");
    if (coreExtract.getInput() != coreReduce.getResult())
      return rewriter.notifyMatchFailure(
          coreExtract, "q8_0 integer-core lane0 extract input must be the "
                       "reduce");

    // ---- The pinned-oracle SeparatedLeftAssoc fold descriptor + shared emit
    // state (fold only; the integer core is emitted op-by-op below). The q8_0
    // TYPED body conforms to [testing/flat-block-dot-fp-fold-oracle.md §1]:
    // ((sumi*d_x)*d_y) as SEPARATE emitc statements (no dx*dy premultiply, no FMA
    // contraction). The MONOLITH keeps SumiTimesScales (`sumi*(d_x*d_y)`), so the
    // typed emit is INTENTIONALLY no longer byte-exact vs monolith here -- the
    // sanctioned gate migration (monolith retires later), q8_0 only. ----
    FlatBlockDotDescriptor descriptor;
    descriptor.decodePrimitive = FlatDecodePrimitive::PlainI8;
    descriptor.foldModel = FlatFoldModel::SeparatedLeftAssoc;
    descriptor.defaultCoreLmul = "m2";
    BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m2");
    FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
        rewriter, descriptor, facts, weightBase, activationBase,
        sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType, opName,
        role);

    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    llvm::StringRef coreLmul =
        llvm::cast<tcrvrvv::VectorType>(weightLoad.getLoaded().getType())
            .getLmul();
    llvm::StringRef wideLmul =
        llvm::cast<tcrvrvv::VectorType>(coreProduct.getResult().getType())
            .getLmul();
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
    unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
    llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
    std::string innerSetvlCallee =
        riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
    int64_t blockLen = qk; // q8_0 whole-block strip.

    // ---- The per-block q8_0 core, parameterized by (ib, blockOffset,
    // forceRobust). Mirrors emitQ40Core: emits address / brick-1 scales / the
    // op-by-op integer core (elided single cover OR robust inner strip loop) and
    // returns the fold inputs; the FOLD is emitted by the scaffold (so an
    // mbf-unroll group emits all cores first, then the folds in strict ascending
    // order). ----
    auto emitQ80Core =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<FlatBlockCore> {
      // Fresh per-core base memo: each unrolled core (blockOffset k) recomputes
      // base + (ib+k)*stride; a memo shared across cores would alias block k>0 to
      // block 0's address.
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value idx = ib;
        if (blockOffset != 0)
          idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                              sizeLit(blockOffset));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };
      mlir::Value xb = blockBaseFor(
          weightLoad.getBuffer(), weightLoad.getBlockIndex(),
          static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
      mlir::Value yb = blockBaseFor(
          activationLoad.getBuffer(), activationLoad.getBlockIndex(),
          static_cast<int64_t>(*activationLoad.getBlockStride()), "block_base_y");
      auto fp16ReadAt = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
        mlir::Value addr = blockBase;
        if (byteOffset && *byteOffset != 0)
          addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                               blockBase, sizeLit(*byteOffset));
        return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                              mlir::ValueRange{addr}, opName, role,
                              llvm::StringRef("fcvt.s.h"));
      };
      mlir::Value dX = fp16ReadAt(
          blockBaseFor(brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                       "block_base_x"),
          brick1.getLhsScaleByteOffset());
      mlir::Value dY = fp16ReadAt(
          blockBaseFor(brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                       "block_base_y"),
          brick1.getRhsScaleByteOffset());

      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // One strip: op-sourced weight+activation i8 loads + plain-i8 signed
      // widening vwmul product + reduce + lane0 extract + assign sumi.
      // chunkOffset is the elided 0 literal or the robust strip induction var;
      // carrySumi seeds lane0 from the sumi lvalue (robust re-strip) or a fresh 0
      // (elided single cover).
      auto emitStrip = [&](mlir::Value chunkOffset, mlir::Value vl,
                           bool carrySumi) -> mlir::LogicalResult {
        auto emitLoadTail = [&](mlir::Value blockBase,
                                int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTail(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[activationLoad.getLoaded()] = emitLoadTail(
            yb, static_cast<int64_t>(*activationLoad.getQuantByteOffset()));
        // The signed widening vwmul, emitted from ITS lhs/rhs load-result
        // operands (via the valueMap) -- byte-exact to the monolith's
        // `__riscv_vwmul_vv_i16m4(vx0, vy0, vl)` (lhs=weight first).
        valueMap[coreProduct.getResult()] = emitOpaqueCall(
            rewriter, loc, i16WideType, mulCallee,
            mlir::ValueRange{valueMap.lookup(coreProduct.getLhs()),
                             valueMap.lookup(coreProduct.getRhs()), vl},
            opName, role);
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value sumiSeed =
                  carrySumi
                      ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                            .getResult()
                      : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                            .getResult();
              return {sumiSeed, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, "__riscv_vmv_x_s_i32m1_i32",
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);
        return mlir::success();
      };

      bool robust = forceRobust || !stripElided;
      if (!robust) {
        // Elided single cover: ONE vsetvl(block_len) caps the active vl at the
        // whole block, ONE strip reduce, no inner loop, no sumi carry.
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        if (mlir::failed(emitStrip(sizeLit(0), vl, /*carrySumi=*/false)))
          return mlir::failure();
      } else {
        // Robust inner strip loop: step = loop-invariant VLMAX, per-chunk active
        // vl = vsetvl(block_len - c), sumi-carrying seed. VLEN-robust re-strip.
        mlir::Value innerVlmax = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        auto innerLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
            /*bodyBuilder=*/nullptr);
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining = rewriter.create<emitc::SubOp>(
                  loc, sizeType, sizeLit(blockLen), c);
              return {remaining};
            });
        if (mlir::failed(emitStrip(c, vl, /*carrySumi=*/true)))
          return mlir::failure();
      }
      return FlatBlockCore{sumiVar.getResult(), dX, dY, mlir::Value(),
                           mlir::Value()};
    };

    // ---- The mbf schedule scaffold (identical to emitFlatBlockDot / emitQ40Core):
    // mbf==1 = single block loop {core; fold}; mbf>1 = a main loop stepping by
    // factor emitting ALL factor cores FIRST then the factor folds in STRICT
    // ascending block order (the fp non-associativity boundary), plus a robust
    // single-block scalar tail over the nb % factor remainder. ----
    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<FlatBlockCore> core =
          emitQ80Core(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                   core->mX, core->sY);
    } else {
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nbMain, factorLit, /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<FlatBlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<FlatBlockCore> core =
              emitQ80Core(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const FlatBlockCore &core : cores)
          emitFlatFold(rewriter, loc, st, core.sumiVar, core.dX, core.dY,
                       core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(
          loc, nbMain, nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<FlatBlockCore> core =
            emitQ80Core(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                     core->mX, core->sY);
      }
    }
  } else {
    // ---- The remaining folds (q4_1 / q5_0 / q5_1) and the step-1-3 skeleton
    // still materialize only the mbf==1 + elided default; any unroll/robust combo
    // is a later step (fail-closed, I7). q8_0 (sumi_times_scales) and q4_0
    // (left_assoc) take the schedule-parameterized branches above. ----
    if (multiBlockFactor != 1 || !stripElided)
      return rewriter.notifyMatchFailure(
          loopBody,
          "multi_block_factor>1 and the robust strip form are currently "
          "materialized only for the left_assoc (q4_0) and sumi_times_scales "
          "(q8_0) flat bodies; the other folds require the "
          "multi_block_factor==1 + strip_elision==elided default (later steps)");

  // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- byte-exact to
  // emitFlatBlockDot:5879-5883 (the no-unroll block loop).
  auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());

    mlir::Block &coreBlock = loopBody.getBody().front();

    // M-FLAT step 5a + W4: recognize the COMPLETE q8_0 typed body in the region
    // and drive the byte-exact FULL-body emit OP-BY-OP from the region ops'
    // operands (NOT套壳 re-derived from the loop-body attrs through the
    // monolithic emitFlatBlockCore). The region carries the FULL per-block chain:
    // brick 1 (the per-block d_x*d_y scale) -> the vector integer core (two
    // per-block i8 loads -> signed widening product -> standalone reduce -> lane0
    // scalar extract into the i32 sumi) -> brick 2 (the computed-scale i32-sumi
    // dequant on that sumi) -> brick 3 (the cross-block fp32 fold) -> yield. The
    // chain is recognized REGION-DRIVEN: the presence of brick 2 marks a full
    // body, and the emit is GATED on the region's actual op WIRING -- a missing
    // brick 2, a fold folded on a non-brick-1 scale, OR a broken/misdirected
    // integer-core link (W4) fails closed here. After the gate, each op is emitted
    // from ITS operands: the load's block_stride/quant_byte_offset drive the
    // address arithmetic, the product reads its lhs/rhs load results, the reduce
    // reads its product, the extract reads its reduce, and brick 1's two fp16
    // reads share the load's per-block base (memoized on the (buffer, block_index)
    // SSA pair). Only the fused fold tail (brick 1's d_x*d_y + brick 2's cast/mul
    // + brick 3's add) collapses COLLECTIVELY into one emitc.expression via the
    // reused emitFlatFold, fed the operand-derived d_x/d_y + sumi. The result is
    // byte-identical to the monolithic q8_0 (verified by golden diff) yet sourced
    // from the region ops, not the attrs.
    tcrvrvv::BlockFp16ScaleProductOp brick1;
    tcrvrvv::BlockComputedScaleDequantOp brick2;
    tcrvrvv::CrossBlockF32AccumulateOp brick3;
    tcrvrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    // W4: the region's vector integer-core chain (two per-block i8 loads ->
    // signed widening product -> standalone reduce -> lane0 scalar extract ->
    // the scalar i32 sumi feeding brick 2). Collected so the full-body gate can
    // check every link against the actual region SSA wiring.
    llvm::SmallVector<tcrvrvv::LoadOp, 2> coreLoads;
    tcrvrvv::WideningProductOp coreProduct;
    tcrvrvv::StandaloneReduceOp coreReduce;
    tcrvrvv::TypedVectorLane0ToScalarExtractOp coreExtract;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<tcrvrvv::BlockFp16ScaleProductOp>(bodyOp))
        brick1 = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::BlockComputedScaleDequantOp>(bodyOp))
        brick2 = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::CrossBlockF32AccumulateOp>(bodyOp))
        brick3 = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::LoadOp>(bodyOp))
        coreLoads.push_back(o);
      else if (auto o = llvm::dyn_cast<tcrvrvv::WideningProductOp>(bodyOp))
        coreProduct = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(bodyOp))
        coreReduce = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedVectorLane0ToScalarExtractOp>(
                       bodyOp))
        coreExtract = o;
    });

    if (brick2) {
      // ---- Full q8_0 body: the REGION-DRIVEN gate (defends against a
      // cosmetic attribute-rederive). Every link is checked against the actual
      // region SSA wiring; any break fails closed so the emit provably tracks
      // the region content, not just the loop-body attrs. ----
      if (!brick1 || !brick3 || !yieldOp)
        return rewriter.notifyMatchFailure(
            loopBody, "full flat block-dot body requires brick 1 (per-block "
                      "scale), brick 2 (computed-scale dequant), brick 3 "
                      "(cross-block fold), and the yield");
      if (brick1.getBlockIndex() != coreBlock.getArgument(0))
        return rewriter.notifyMatchFailure(
            brick1, "brick 1 block_index must be the loop induction variable");
      if (brick2.getComputedScale() != brick1.getResult())
        return rewriter.notifyMatchFailure(
            brick2, "brick 2 computed_scale must be brick 1's per-block scale");
      if (brick3.getAcc() != coreBlock.getArgument(1))
        return rewriter.notifyMatchFailure(
            brick3, "brick 3 acc must be the loop-carried accumulator");
      if (brick3.getTerm() != brick2.getResult())
        return rewriter.notifyMatchFailure(
            brick3, "brick 3 term must be brick 2's per-block dequant term");
      if (yieldOp.getAccNext() != brick3.getResult())
        return rewriter.notifyMatchFailure(
            yieldOp, "loop yield acc_next must be brick 3's fold result");

      // fold_model COLLISION tiebreaker (M-FLAT q5_1 = cohort LAST cell): q4_1 AND
      // q5_1 BOTH stamp fold_model="scale_plus_min", so fold_model is no longer a
      // unique key. q5_1 = q5_0's five-bit integer core (five-bit product + qh
      // brick, offset-bias OFF) UNIONed with q4_1's MIN term. Detect by op
      // IDENTITY -- a five-bit product + qh brick under the scale_plus_min fold =>
      // q5_1; else the fold_model chain below dispatches q8_0/q4_0/q4_1/q5_0
      // byte-unchanged (q5_0's scales_times_sumi + q4_1's min-only scale_plus_min
      // both fail this predicate).
      tcrvrvv::FiveBitOffsetBinaryXI8ProductOp q51FiveBitProduct;
      loopBody.getBody().walk([&](tcrvrvv::FiveBitOffsetBinaryXI8ProductOp o) {
        q51FiveBitProduct = o;
      });
      tcrvrvv::BlockFiveBitQhSourceOp q51QhBrick;
      loopBody.getBody().walk(
          [&](tcrvrvv::BlockFiveBitQhSourceOp o) { q51QhBrick = o; });
      tcrvrvv::BlockFp16MinProductOp q51MinBrick;
      loopBody.getBody().walk(
          [&](tcrvrvv::BlockFp16MinProductOp o) { q51MinBrick = o; });
      const bool isQ51Body = loopBody.getFoldModel() == "scale_plus_min" &&
                             q51FiveBitProduct && q51QhBrick;

      // Fold-tree dispatch. Both branches emit INTO the block loop body and FALL
      // THROUGH to the shared post-loop `*s = sumf` store below (do NOT return
      // here -- a return would skip the store and ship a kernel that computes but
      // never writes). q4_0's left_assoc fold DESCRIPTOR-DRIVES the SHARED
      // emitFlatBlockCore + emitFlatFold (byte-identical to the monolithic q4_0
      // mbf1 body); q8_0's sumi_times_scales takes the op-by-op W4 emit. Any other
      // fold tree fails closed (I7).
      if (isQ51Body) {
        // ---- q5_1 (M-FLAT cohort LAST cell): the UNION of q5_0's five-bit
        // offset-binary integer core (five-bit product + qh 5th-bit brick) and
        // q4_1's Family-B MIN correction, folded through the ScalePlusMin tree.
        // The SINGLE arithmetic delta vs q5_0 is applyOffsetBias=FALSE (the `-16`
        // bias lives in the per-block MIN scale, so NO `vsub 16` is emitted). The
        // MIN reads (m_x/s_y) sit in the q4_1 slot RIGHT AFTER dX/dY and BEFORE
        // the qh halves -- byte-identical to the monolithic q5_1
        // (FiveBitOffsetBinary / half-block / ScalePlusMin, applyOffsetBias=false,
        // m1, elided, mbf 1) emitFlatBlockCore read order
        // (dX,dY,mX,sY,qhLow16,qhHigh16). The decode + fold + qh source + MIN are
        // all selected from the WALKED op identity (anti-bypass gates product + qh
        // + min), not the fold_model string. ----
        tcrvrvv::FiveBitOffsetBinaryXI8ProductOp fiveBitProduct = q51FiveBitProduct;
        tcrvrvv::BlockFiveBitQhSourceOp qhBrick = q51QhBrick;
        tcrvrvv::BlockFp16MinProductOp minBrick = q51MinBrick;
        if (!fiveBitProduct || !qhBrick || !minBrick || coreLoads.size() != 3 ||
            !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q5_1 flat block-dot body requires the region integer core: "
              "three per-block loads (a u8 packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric five-bit offset-binary packed-i4 x i8 "
              "product, a lane0 scalar extract, the per-block qh-source brick, "
              "and the per-block MIN brick");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");
        // The KEY Family-B link (I7): brick 2's min_term must be the MIN brick's
        // per-block m_x*s_y product. q5_1's ScalePlusMin fold REQUIRES it (INVERTS
        // q5_0's "must NOT carry a min_term").
        if (brick2.getMinTerm() != minBrick.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "q5_1 brick 2 min_term must be the per-block MIN brick's "
                      "m_x*s_y correction product");
        // The KEY five-bit link (I7): the product's qh_source operand must be the
        // per-block qh brick's gate-only token.
        if (fiveBitProduct.getQhSource() != qhBrick.getResult())
          return rewriter.notifyMatchFailure(
              fiveBitProduct, "q5_1 five-bit product qh_source must be the "
                              "per-block qh-source brick's gate-only token");

        // Follow the five-bit product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight (u8), q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh.
        auto weightLoad =
            fiveBitProduct.getWeight().getDefiningOp<tcrvrvv::LoadOp>();
        auto lowLoad =
            fiveBitProduct.getActivationLow().getDefiningOp<tcrvrvv::LoadOp>();
        auto highLoad =
            fiveBitProduct.getActivationHigh().getDefiningOp<tcrvrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride())
          return rewriter.notifyMatchFailure(
              loopBody,
              "q5_1 five-bit product operands must be the region's per-block "
              "loads: a u8 packed-i4 weight load off the weight ABI buffer + two "
              "plain-i8 q8 (low/high) activation loads off the activation ABI "
              "buffer, each carrying a block_stride + quant_byte_offset");

        if (!coreReduce ||
            coreReduce.getInput() != fiveBitProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q5_1 integer-core reduce input must be the five-bit product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q5_1 integer-core lane0 extract input must be the "
                           "reduce");

        // The ScalePlusMin fold descriptor with applyOffsetBias=FALSE (the ONE
        // arithmetic delta vs q5_0). Only descriptor.foldModel + the shared state
        // are consumed by emitFlatFold; the integer core + the qh + MIN harvest
        // are emitted op-by-op below.
        FlatBlockDotDescriptor descriptor;
        descriptor.decodePrimitive = FlatDecodePrimitive::FiveBitOffsetBinary;
        descriptor.foldModel = FlatFoldModel::ScalePlusMin;
        descriptor.defaultCoreLmul = "m1";
        descriptor.applyOffsetBias = false;
        BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m1");
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // ONE shared per-block base memo across the merged sub-paths (brick 1's +
        // the MIN brick's + the qh brick's reads and the per-block loads all name
        // the same (%buffer, %block_index) SSA pair -> the memo must hit once; a
        // double-declared base re-emits the `base + ib*stride` and byte-diffs).
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1's two per-block fp16 -> f32 SCALE reads (d_x/d_y at offset 0).
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());

        // The MIN brick's m_x/s_y fp16 reads, emitted RIGHT AFTER dX/dY and BEFORE
        // the qh halves (byte-exact to the q5_1 monolith read order
        // dX,dY,mX,sY,qhLow16,qhHigh16). Off the SHARED bases (memo hits) at the
        // MIN brick's OWN byte offsets -- anti-bypass: mutating the min brick's
        // operands/offsets changes these emitted addresses.
        mlir::Value mX = fp16ReadAt(
            blockBaseFor(
                minBrick.getLhsMinBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            minBrick.getLhsMinByteOffset());
        mlir::Value sY = fp16ReadAt(
            blockBaseFor(
                minBrick.getRhsSumBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            minBrick.getRhsSumByteOffset());

        // The qh field's TWO aligned 16-bit halves, read AFTER m_x/s_y and BEFORE
        // the sumi decl. Off the SHARED weight base (memo hit) at the qh brick's
        // OWN qh_byte_offset -- anti-bypass on the qh brick's qh_base/qh_byte_offset;
        // the descriptor qh offset is NEVER read here. NOT fp16ReadAt: the qh read
        // is a raw `(uint16_t)*(const uint16_t *)` call to a u32.
        mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
        llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
        mlir::Value qhBase = blockBaseFor(
            qhBrick.getQhBase(), qhBrick.getBlockIndex(),
            static_cast<int64_t>(qhBrick.getBlockStride().value_or(0)),
            "block_base_x");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_field"));
        auto u16ReadAt = [&](int64_t byteOffset) -> mlir::Value {
          mlir::Value ptr = rewriter.create<emitc::AddOp>(
              loc, qhBase.getType(), qhBase, sizeLit(byteOffset));
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                           u16ReadCallee, mlir::ValueRange{ptr})
              .getResult(0);
        };
        int64_t qhOffset =
            static_cast<int64_t>(qhBrick.getQhByteOffset().value_or(0));
        mlir::Value qhLow16 = u16ReadAt(qhOffset);
        mlir::Value qhHigh16 = u16ReadAt(qhOffset + 2);

        // The i32 sumi lvalue.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2). coreLmul is the
        // LOAD result LMUL (u8m1 for the weight; the vsetvl spelling is m1).
        auto loadVecType =
            llvm::cast<tcrvrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(qk / 2)};
            });

        // HETEROGENEOUS loads: the WEIGHT strip is loaded U8 while the two q8
        // activation halves stay I8. The weight/activation quant offsets DIVERGE
        // (weight@8, acts@4/20) -- read straight off the load ops.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type u8CoreType =
            emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        mlir::Type u8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTailI8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        auto emitLoadTailU8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTailU8(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTailI8(yb, static_cast<int64_t>(*lowLoad.getQuantByteOffset()));
        valueMap[highLoad.getLoaded()] = emitLoadTailI8(
            yb, static_cast<int64_t>(*highLoad.getQuantByteOffset()));

        // The five-bit offset-binary nibble+qh decode + asymmetric widening
        // product, emitted from the product op's OWN weight/low/high operands (via
        // the valueMap) + the re-read qh halves + chunkOffset 0. applyOffsetBias
        // =FALSE is the ONE arithmetic delta vs q5_0: NO `vsub 16` is emitted (the
        // bias lives in the per-block MIN scale, folded through m_x*s_y).
        auto prodVecType = llvm::cast<tcrvrvv::VectorType>(
            fiveBitProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::Type u16WideType =
            emitc::OpaqueType::get(ctx, ("vuint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitFiveBitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(fiveBitProduct.getWeight()),
                valueMap.lookup(fiveBitProduct.getActivationLow()),
                valueMap.lookup(fiveBitProduct.getActivationHigh()), qhLow16,
                qhHigh16, chunkOffset, vl, i8CoreType, u8CoreType, u16WideType,
                i16WideType, coreLmul, wideLmul, 16, wideLmul, "i16", opName,
                role, /*applyOffsetBias=*/false);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[fiveBitProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand: seed a FRESH
        // literal-0 lane (per-block, no sumi carry), vwredsum the product, pull
        // lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // brick 1 ((d_x*d_y)*sumi) + the MIN brick (m_x*s_y) + brick 3 (sumf +
        // term) fold COLLECTIVELY into the one fused emitc.expression
        // (ScalePlusMin), fed the operand-derived d_x/d_y/m_x/s_y + the sumi
        // lvalue. The m_x*s_y mul is recomputed INSIDE the fused expression, so
        // the min brick's f32 result is gate-only.
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY, mX, sY);
      } else if (loopBody.getFoldModel() == "left_assoc") {
        // ---- q4_0 half-block asymmetric offset-binary packed-i4 x i8 core.
        // Lightweight region gate (I7 fail-closed): the region must carry the
        // asymmetric packed-i4 product, the THREE per-block loads (one packed-i4
        // weight + two plain-i8 q8 halves), and the lane0 extract feeding brick
        // 2's sumi. Then the descriptor-driven emit reuses the SAME
        // emitFlatBlockCore + emitFlatFold the monolithic q4_0 uses, so the body
        // is byte-identical (the OffsetBinaryNibble decode + LeftAssoc fold). ----
        tcrvrvv::PackedI4OffsetBinaryXI8ProductOp packedProduct;
        loopBody.getBody().walk(
            [&](tcrvrvv::PackedI4OffsetBinaryXI8ProductOp o) {
              packedProduct = o;
            });
        if (!packedProduct || coreLoads.size() != 3 || !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q4_0 flat block-dot body requires the region integer core: "
              "three per-block i8 loads (a packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric offset-binary packed-i4 x i8 product, and "
              "a lane0 scalar extract");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");

        // ---- W4 (q4_0 anti-bypass): OP-BY-OP OPERAND-DRIVEN packed-i4 core. The
        // integer core is NO LONGER re-derived from the loop-body attrs through
        // the monolithic emitFlatBlockCore套壳: the packed-i4 product is lowered
        // from ITS OWN weight/activation_low/activation_high operands (via the
        // valueMap), so mutating any product operand changes the emitted bytes.
        // Pre-W4 the packed product was收 as PRESENCE only and the core was rebuilt
        // from the LOADS' descriptor offsets, so swapping the product's act_high /
        // act_low / weight operand emitted the SAME bytes (NOT_GENUINE). Each of
        // the THREE loads is emitted from ITS load op's buffer/block_stride/
        // quant_byte_offset; the packed decode+widening product reads the three
        // load results the product NAMES; the reduce reads its product operand; the
        // lane0 extract reads its reduce operand; brick 1's two fp16 reads share the
        // per-block base. The decode arithmetic routes to the SAME
        // emitOffsetBinaryDecodeProductValue (vxor 0x88 / vsll / vsra sign-extend /
        // vwmul low / vwmacc high) the monolith's OffsetBinaryNibble strip runs, and
        // emitFlatFold emits the LeftAssoc fp32 fold -- so the body stays
        // byte-identical to the monolithic q4_0 mbf1 instance, only sourced from the
        // region ops. ----

        // Follow the packed product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight, q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh. Each must be a
        // per-block load reading the matching ABI buffer with a quant_byte_offset.
        auto weightLoad =
            packedProduct.getWeight().getDefiningOp<tcrvrvv::LoadOp>();
        auto lowLoad =
            packedProduct.getActivationLow().getDefiningOp<tcrvrvv::LoadOp>();
        auto highLoad =
            packedProduct.getActivationHigh().getDefiningOp<tcrvrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride())
          return rewriter.notifyMatchFailure(
              loopBody,
              "q4_0 packed-i4 product operands must be the region's per-block "
              "loads: a packed-i4 weight load off the weight ABI buffer + two "
              "plain-i8 q8 (low/high) activation loads off the activation ABI "
              "buffer, each carrying a block_stride + quant_byte_offset");

        // The standalone reduce consumes the packed product; the lane0 extract
        // consumes the reduce (brick 2's sumi = the extract is checked above). Any
        // break fails closed so the emit provably tracks the region chain.
        if (!coreReduce || coreReduce.getInput() != packedProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q4_0 integer-core reduce input must be the packed-i4 product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q4_0 integer-core lane0 extract input must be the "
                           "reduce");

        // The LeftAssoc fold descriptor. Only descriptor.foldModel + the shared
        // state (sumfVar / floatType / i32Type) are consumed by emitFlatFold; the
        // integer core is emitted op-by-op below, NOT via emitFlatBlockCore.
        FlatBlockDotDescriptor descriptor;
        descriptor.decodePrimitive = FlatDecodePrimitive::OffsetBinaryNibble;
        descriptor.foldModel = FlatFoldModel::LeftAssoc;
        descriptor.defaultCoreLmul = "m1";
        BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m1");
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // Shared per-block base memo (byte-exact to the monolith's single
        // blockBaseValue): brick 1's fp16 reads + the per-block i8 loads that name
        // the same (%buffer, %block_index) SSA pair share ONE `base + ib*stride`.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        // block_base_x FROM the weight load; block_base_y FROM the low q8 load (the
        // high q8 load names the SAME activation base + block_index -> memo hit).
        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1: the two per-block fp16 -> f32 scale reads off the SHARED base
        // (memo hits, no re-emit), byte-exact to the monolith's fp16ReadAt.
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());

        // The i32 sumi lvalue: `int32_t sumi; sumi = 0;` (byte-exact BQL:5735).
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2) -- q4_0's nibble
        // half-block strip is qk/2 bytes (NOT the whole qk q8_0 uses). coreLmul is
        // the LOAD result LMUL (i8m1).
        auto loadVecType =
            llvm::cast<tcrvrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(qk / 2)};
            });

        // The THREE per-block i8 loads, emitted in the monolith's order (weight ->
        // low q8 -> high q8), each from ITS load op's block base + quant_byte_offset
        // (+ the SHARED 0 chunk literal) + i8* cast + vle8 (byte-exact to
        // chunkPtr+loadI8). Keyed in the valueMap on the load result so the packed
        // product below reads them by the operand the product NAMES.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTail = [&](mlir::Value blockBase,
                                int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTail(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTail(yb, static_cast<int64_t>(*lowLoad.getQuantByteOffset()));
        valueMap[highLoad.getLoaded()] = emitLoadTail(
            yb, static_cast<int64_t>(*highLoad.getQuantByteOffset()));

        // The packed-i4 offset-binary decode + asymmetric widening product,
        // emitted from the packed product op's OWN weight/low/high operands (via
        // the valueMap) -- the SAME emitOffsetBinaryDecodeProductValue arithmetic
        // (vxor 0x88 / vsll / vsra sign-extend / vwmul low / vwmacc high) the
        // monolith's OffsetBinaryNibble strip reduce runs, so byte-identical while
        // the operands stay op-sourced. wideLmul is the product RESULT LMUL (i16m2).
        auto prodVecType =
            llvm::cast<tcrvrvv::VectorType>(packedProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(packedProduct.getWeight()),
                valueMap.lookup(packedProduct.getActivationLow()),
                valueMap.lookup(packedProduct.getActivationHigh()), vl,
                i8CoreType, i16WideType, "i8", coreLmul, 16, wideLmul, "i16",
                opName, role);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[packedProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand (byte-exact
        // BQL:5711-5766): seed a FRESH literal-0 lane (per-block, no sumi carry),
        // vwredsum the product, pull lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // brick 1 ((float)sumi * d_x) * d_y (LeftAssoc) + brick 3 (sumf + term)
        // fold COLLECTIVELY into the one fused emitc.expression, fed the
        // operand-derived d_x/d_y + the sumi lvalue. emitFlatFold's fold tree is
        // the gated fold_model attr.
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY,
                     /*mX=*/mlir::Value(), /*sY=*/mlir::Value());
      } else if (loopBody.getFoldModel() == "sumi_times_scales") {

      // The q8_0 (plain_i8 / whole-block) descriptor rebuilt from the loop-body
      // scheduling facts. The int8 quant payload sits past the 2-byte fp16 scale
      // header (quantOffset = stride - qk); q8_0 carries no qh / min / codebook.
      // (This else-branch full-body path is superseded by isQ80ScheduleParam for
      // any brick-2 body; kept in sync with the pinned SeparatedLeftAssoc oracle.)
      FlatBlockDotDescriptor descriptor;
      descriptor.decodePrimitive = FlatDecodePrimitive::PlainI8;
      descriptor.foldModel = FlatFoldModel::SeparatedLeftAssoc;
      descriptor.defaultCoreLmul = "m2";
      descriptor.qk = qk;
      descriptor.weightStride = loopBody.getWeightBlockStride();
      descriptor.activationStride = loopBody.getActivationBlockStride();
      descriptor.blockLen = qk;
      descriptor.quantOffset = descriptor.weightStride - qk;
      descriptor.activationQuantOffset = descriptor.activationStride - qk;

      // ---- W4: the REGION-DRIVEN integer-core chain gate. The full-body emit is
      // driven OP-BY-OP from these ops' operands (below); this gate first pins the
      // region as actually CARRYING the chain -- two per-block i8 loads -> signed
      // widening product -> standalone reduce -> lane0 scalar extract -> the
      // scalar i32 sumi feeding brick 2. Every link is checked against the
      // region SSA wiring; deleting or misdirecting ANY link fails closed here,
      // so the emit provably tracks the region content (pre-W4 the chain was
      // ignored and a broken/missing chain still emitted). ----
      if (coreLoads.size() != 2 || !coreProduct || !coreReduce || !coreExtract)
        return rewriter.notifyMatchFailure(
            loopBody,
            "full flat block-dot body requires the region integer core: two "
            "per-block i8 loads, a signed widening product, a standalone "
            "reduce, and a lane0 scalar extract");

      // Identify the weight vs activation per-block load by ABI buffer.
      tcrvrvv::LoadOp weightLoad, activationLoad;
      for (tcrvrvv::LoadOp ld : coreLoads) {
        if (ld.getBuffer() == loopBody.getWeightBase())
          weightLoad = ld;
        else if (ld.getBuffer() == loopBody.getActivationBase())
          activationLoad = ld;
      }
      if (!weightLoad || !activationLoad)
        return rewriter.notifyMatchFailure(
            loopBody, "integer-core loads must read the weight and activation "
                      "ABI buffers of the loop body");

      // Both loads are per-block, striding on the loop induction variable, with
      // the block stride / quant offset the descriptor scheduled (byte-exact to
      // the monolith's blockBaseValue+loadI8).
      for (auto [ld, stride, quantOff] :
           {std::make_tuple(weightLoad, descriptor.weightStride,
                            descriptor.quantOffset),
            std::make_tuple(activationLoad, descriptor.activationStride,
                            descriptor.activationQuantOffset)}) {
        if (ld.getBlockIndex() != coreBlock.getArgument(0))
          return rewriter.notifyMatchFailure(
              ld, "integer-core load block_index must be the loop induction "
                  "variable");
        if (ld.getBlockStride() != stride)
          return rewriter.notifyMatchFailure(
              ld, "integer-core load block_stride must match the scheduled "
                  "block stride");
        if (ld.getQuantByteOffset() != quantOff)
          return rewriter.notifyMatchFailure(
              ld, "integer-core load quant_byte_offset must match the "
                  "scheduled quant offset");
      }

      // The signed widening product multiplies the two per-block load results
      // (operand order free).
      if (coreProduct.getKind() != "signed_widening_product")
        return rewriter.notifyMatchFailure(
            coreProduct, "integer-core widening product must be a signed "
                         "widening product");
      mlir::Value pl = coreProduct.getLhs(), pr = coreProduct.getRhs();
      mlir::Value wLoaded = weightLoad.getLoaded(),
                  aLoaded = activationLoad.getLoaded();
      if (!((pl == wLoaded && pr == aLoaded) ||
            (pl == aLoaded && pr == wLoaded)))
        return rewriter.notifyMatchFailure(
            coreProduct, "integer-core widening product operands must be the "
                         "two per-block load results");

      // The standalone reduce consumes the widening product.
      if (coreReduce.getInput() != coreProduct.getResult())
        return rewriter.notifyMatchFailure(
            coreReduce,
            "integer-core reduce input must be the widening product");

      // The lane0 extract consumes the reduce.
      if (coreExtract.getInput() != coreReduce.getResult())
        return rewriter.notifyMatchFailure(
            coreExtract, "integer-core lane0 extract input must be the reduce");

      // The KEY link: brick 2's scalar i32 sumi must be the extracted lane0
      // scalar (the integer core -> scalar fold handoff), NOT a direct sumi.
      if (brick2.getSumi() != coreExtract.getResult())
        return rewriter.notifyMatchFailure(
            brick2, "brick 2 sumi must be the integer-core lane0 extract "
                    "result");

      // ---- W4: OP-BY-OP OPERAND-DRIVEN integer-core + scale emit. The full-body
      // integer core is NO LONGER re-derived from the loop-body attrs through the
      // monolithic emitFlatBlockCore套壳: every region op is lowered from ITS OWN
      // operands/attrs. The per-block base address flows from the LOAD op's
      // block_stride/quant_byte_offset (the address arithmetic tracks the load
      // operand, not the descriptor); the widening product reads its lhs/rhs load
      // results through the valueMap; the reduce reads its product operand; the
      // lane0 extract reads its reduce operand; brick 1's two fp16 reads read the
      // SHARED per-block base; and brick 1's d_x*d_y (+ brick 2 cast/mul + brick 3
      // add) fold COLLECTIVELY into the one fused emitc.expression. The emit stays
      // byte-identical to the monolithic q8_0 (plain_i8 / whole-block /
      // SumiTimesScales, m2, elided, mbf 1) instance (verified by golden diff), but
      // is now sourced from the region ops. Only emitFlatFold (the fused-tail
      // assembler, fed operand-derived d_x/d_y + the sumi lvalue) is reused for the
      // brick 1/2/3 collapse; the fold_model selecting the tree is the gated attr.
      BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m2");
      FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
          rewriter, descriptor, facts, weightBase, activationBase,
          sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
          opName, role);

      mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
      mlir::Value ib = blockLoop.getInductionVar();

      // Shared per-block base, key = (buffer SSA value, block_index SSA value).
      // brick 1's fp16-scale reads and the per-block i8 load BOTH name the same
      // (%buffer, %block_index) SSA pair, so ONE `base + ib*stride` (+ one step
      // comment) serves both -- byte-exact to the monolith's single blockBaseValue
      // build; emitting it twice would be a byte-diff.
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // block_base_x / block_base_y FROM the two per-block LOAD ops: the load's
      // block_stride drives the address mul (the operand-flow real gate).
      mlir::Value xb = blockBaseFor(
          weightLoad.getBuffer(), weightLoad.getBlockIndex(),
          static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
      mlir::Value yb = blockBaseFor(
          activationLoad.getBuffer(), activationLoad.getBlockIndex(),
          static_cast<int64_t>(*activationLoad.getBlockStride()), "block_base_y");

      // brick 1: the two per-block fp16 -> f32 scale reads, off the SHARED base
      // (memo hit, no re-emit). d_x = read(block_base_x + lhs_scale_byte_offset),
      // d_y = read(block_base_y + rhs_scale_byte_offset), byte-exact to fp16ReadAt.
      auto fp16ReadAt = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
        mlir::Value addr = blockBase;
        if (byteOffset && *byteOffset != 0)
          addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                               blockBase, sizeLit(*byteOffset));
        return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                              mlir::ValueRange{addr}, opName, role,
                              llvm::StringRef("fcvt.s.h"));
      };
      mlir::Value dX = fp16ReadAt(
          blockBaseFor(
              brick1.getLhsScaleBase(), brick1.getBlockIndex(),
              static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
              "block_base_x"),
          brick1.getLhsScaleByteOffset());
      mlir::Value dY = fp16ReadAt(
          blockBaseFor(
              brick1.getRhsScaleBase(), brick1.getBlockIndex(),
              static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
              "block_base_y"),
          brick1.getRhsScaleByteOffset());

      // The i32 sumi lvalue: `int32_t sumi; sumi = 0;` -- the mutable target the
      // lane0 extract reassigns and the fold loads (byte-exact BQL:5735-5740).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The inner block-capped vl: ONE __riscv_vsetvl_e8<lmul>(qk) threaded as the
      // bodyVL to every core intrinsic. This is the ONE spot NOT sourced from the
      // region op operands -- the region ops carry the OUTER scope %vl, but the
      // byte-exact core caps the active vl at the qk sub-block. coreLmul is derived
      // from the LOAD's result vector type.
      auto loadVecType =
          llvm::cast<tcrvrvv::VectorType>(weightLoad.getLoaded().getType());
      llvm::StringRef coreLmul = loadVecType.getLmul();
      unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
      llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
      mlir::Value vl = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(qk)};
          });

      // The two per-block i8 loads, each from ITS load op's block base + the
      // load's quant_byte_offset (+ the SHARED 0 chunk literal) + i8* cast + vle8.
      // The chunk-offset literal `0` is built ONCE and shared by both loads (the
      // monolith's single sizeLit(0) chunkOffset); only the quant offset literal
      // is fresh per load. The vle spelling is derived from the LOAD's LMUL.
      mlir::Type i8CoreType =
          emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
      mlir::Type i8PtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      mlir::Value chunkOffset = sizeLit(0);
      auto emitLoadTail = [&](mlir::Value blockBase,
                              int64_t quantOff) -> mlir::Value {
        mlir::Value withFixed = rewriter.create<emitc::AddOp>(
            loc, blockBase.getType(), blockBase, sizeLit(quantOff));
        mlir::Value full = rewriter.create<emitc::AddOp>(
            loc, blockBase.getType(), withFixed, chunkOffset);
        mlir::Value ptr =
            rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
        return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      valueMap[weightLoad.getLoaded()] = emitLoadTail(
          xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
      valueMap[activationLoad.getLoaded()] = emitLoadTail(
          yb, static_cast<int64_t>(*activationLoad.getQuantByteOffset()));

      // The signed widening product, emitted from ITS lhs/rhs load-result
      // operands (via the valueMap) with the vwmul callee + widened type derived
      // from the RESULT vector type -- byte-exact to the monolith
      // `__riscv_vwmul_vv_i16m4(vx0, vy0, vl)` (lhs=weight first). The inner vl is
      // the bodyVL. emitWideningProduct would produce the identical call but stamp
      // the widening_product op's own provenance verbatim (a byte-diff vs the
      // monolith's loop-body provenance); hand-emitting here keeps the whole body
      // byte-identical while the operands + callee stay op-sourced.
      auto prodVecType =
          llvm::cast<tcrvrvv::VectorType>(coreProduct.getResult().getType());
      llvm::StringRef wideLmul = prodVecType.getLmul();
      mlir::Type i16WideType =
          emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
      std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
      valueMap[coreProduct.getResult()] = emitOpaqueCall(
          rewriter, loc, i16WideType, mulCallee,
          mlir::ValueRange{valueMap.lookup(coreProduct.getLhs()),
                           valueMap.lookup(coreProduct.getRhs()), vl},
          opName, role);

      // Reduce + lane0 extract, each from ITS op's input operand. The reduce seeds
      // a FRESH literal-0 lane (per-block, no sumi carry) then vwredsum's the
      // product; the extract pulls lane0 into the mutable sumi lvalue (byte-exact
      // BQL:5711-5766). The reduce/extract callees derive from the vector types.
      mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, seedCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, sizeLit(1)};
          });
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red = emitOpaqueCall(
          rewriter, loc, i32m1Type, reduceCallee,
          mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
          opName, role);
      valueMap[coreReduce.getResult()] = red;
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      mlir::Value extractVal = emitOpaqueCall(
          rewriter, loc, i32Type, extractCallee,
          mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
          role);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumi", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

      // brick 1 (d_x*d_y) + brick 2 ((float)sumi*scale) + brick 3 (acc+term) fold
      // COLLECTIVELY into the one fused emitc.expression (the single C statement
      // `sumf + (float)sumi * (d_x*d_y)`), fed the operand-derived d_x/d_y (brick 1
      // reads) + the sumi lvalue (from the extract). The acc operand (brick 3) is
      // the loop-carried sumf lvalue that emitFlatFold loads before the expression.
      emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY,
                   /*mX=*/mlir::Value(), /*sY=*/mlir::Value());
      } else if (loopBody.getFoldModel() == "scale_plus_min") {
        // ---- q4_1 half-block asymmetric UNSIGNED-nibble packed-i4 x i8 core +
        // Family-B MIN correction. This is the q4_0 left_assoc op-by-op emit with
        // FOUR deltas: (1) the weight strip is loaded UNSIGNED (u8 / vle8_v_u8m1),
        // (2) the integer core routes to emitUnsignedNibbleDecodeProductValue
        // (vand 0x0F / vsrl 0x04 + reinterpret + vwmul/vwmacc) instead of the
        // offset-binary decode, (3) a per-block MIN brick reads m_x/s_y (fp16 at
        // byte offset 2 off the SHARED block bases), and (4) the fold is the
        // ScalePlusMin tree fed the harvested m_x/s_y. Byte-identical to the
        // monolithic q4_1 mbf1/elided instance; the decode + fold are selected
        // from the WALKED op identity, not the fold_model string. ----
        tcrvrvv::UnsignedNibbleXI8ProductOp unsignedProduct;
        loopBody.getBody().walk(
            [&](tcrvrvv::UnsignedNibbleXI8ProductOp o) { unsignedProduct = o; });
        tcrvrvv::BlockFp16MinProductOp minBrick;
        loopBody.getBody().walk(
            [&](tcrvrvv::BlockFp16MinProductOp o) { minBrick = o; });
        if (!unsignedProduct || !minBrick || coreLoads.size() != 3 ||
            !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q4_1 flat block-dot body requires the region integer core: "
              "three per-block loads (a u8 packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric unsigned-nibble packed-i4 x i8 product, a "
              "lane0 scalar extract, and the per-block MIN brick");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");
        // The KEY Family-B link (I7 fail-closed): brick 2's min_term must be the
        // MIN brick's per-block m_x*s_y product (the scale_plus_min fold's second
        // product). Deleting or misdirecting the min brick fails closed here.
        if (brick2.getMinTerm() != minBrick.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 min_term must be the per-block MIN brick's "
                      "m_x*s_y correction product");

        // Follow the unsigned product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight (u8), q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh.
        auto weightLoad =
            unsignedProduct.getWeight().getDefiningOp<tcrvrvv::LoadOp>();
        auto lowLoad =
            unsignedProduct.getActivationLow().getDefiningOp<tcrvrvv::LoadOp>();
        auto highLoad =
            unsignedProduct.getActivationHigh().getDefiningOp<tcrvrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride())
          return rewriter.notifyMatchFailure(
              loopBody,
              "q4_1 unsigned-nibble product operands must be the region's "
              "per-block loads: a u8 packed-i4 weight load off the weight ABI "
              "buffer + two plain-i8 q8 (low/high) activation loads off the "
              "activation ABI buffer, each carrying a block_stride + "
              "quant_byte_offset");

        if (!coreReduce ||
            coreReduce.getInput() != unsignedProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q4_1 integer-core reduce input must be the unsigned-nibble "
              "product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q4_1 integer-core lane0 extract input must be the "
                           "reduce");

        // The ScalePlusMin fold descriptor. Only descriptor.foldModel + the
        // shared state are consumed by emitFlatFold; the integer core + the min
        // harvest are emitted op-by-op below, NOT via emitFlatBlockCore.
        FlatBlockDotDescriptor descriptor;
        descriptor.decodePrimitive = FlatDecodePrimitive::UnsignedNibble;
        descriptor.foldModel = FlatFoldModel::ScalePlusMin;
        descriptor.defaultCoreLmul = "m1";
        BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m1");
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // Shared per-block base memo (byte-exact to the monolith's single
        // blockBaseValue): brick 1's + the MIN brick's fp16 reads and the
        // per-block loads that name the same (%buffer, %block_index) SSA pair
        // share ONE `base + ib*stride`.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1's two per-block fp16 -> f32 SCALE reads (d_x/d_y at offset 0),
        // off the SHARED base (memo hits, no re-emit), byte-exact to fp16ReadAt.
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());

        // The MIN brick's m_x/s_y fp16 reads, emitted RIGHT AFTER brick 1's dX/dY
        // and BEFORE the sumi decl (byte-exact to the monolith's dX,dY,mX,sY read
        // order). Off the SHARED bases (memo hits) at the MIN brick's OWN byte
        // offsets -- mutating the min brick's operands/offsets changes these
        // emitted addresses -> changes the bytes (anti-bypass); the descriptor min
        // offsets are NEVER read in this path.
        mlir::Value mX = fp16ReadAt(
            blockBaseFor(
                minBrick.getLhsMinBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            minBrick.getLhsMinByteOffset());
        mlir::Value sY = fp16ReadAt(
            blockBaseFor(
                minBrick.getRhsSumBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            minBrick.getRhsSumByteOffset());

        // The i32 sumi lvalue.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2). coreLmul is the
        // LOAD result LMUL (u8m1 for the weight; the vsetvl spelling is m1).
        auto loadVecType =
            llvm::cast<tcrvrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(qk / 2)};
            });

        // HETEROGENEOUS loads: the WEIGHT strip is loaded U8 (const uint8_t* /
        // __riscv_vle8_v_u8m1) while the two q8 activation halves stay I8 --
        // matching the monolith UnsignedNibble case (loadU8 weight + loadI8
        // y0/y1). If the weight were emitted i8, the u8 vand/vsrl callees below
        // would mismatch the operand type.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type u8CoreType =
            emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        mlir::Type u8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTailI8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        auto emitLoadTailU8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTailU8(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTailI8(yb, static_cast<int64_t>(*lowLoad.getQuantByteOffset()));
        valueMap[highLoad.getLoaded()] = emitLoadTailI8(
            yb, static_cast<int64_t>(*highLoad.getQuantByteOffset()));

        // The unsigned-nibble decode + asymmetric widening product, emitted from
        // the product op's OWN weight/low/high operands (via the valueMap) -- the
        // SAME emitUnsignedNibbleDecodeProductValue arithmetic the monolith's
        // UnsignedNibble strip reduce runs, so byte-identical while op-sourced.
        auto prodVecType = llvm::cast<tcrvrvv::VectorType>(
            unsignedProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitUnsignedNibbleDecodeProductValue(
                rewriter, loc, valueMap.lookup(unsignedProduct.getWeight()),
                valueMap.lookup(unsignedProduct.getActivationLow()),
                valueMap.lookup(unsignedProduct.getActivationHigh()), vl,
                i8CoreType, u8CoreType, i16WideType, coreLmul, 16, wideLmul,
                "i16", opName, role);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[unsignedProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand: seed a FRESH
        // literal-0 lane (per-block, no sumi carry), vwredsum the product, pull
        // lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // brick 1 ((d_x*d_y)*sumi) + the MIN brick (m_x*s_y) + brick 3 (sumf +
        // term) fold COLLECTIVELY into the one fused emitc.expression, fed the
        // operand-derived d_x/d_y/m_x/s_y + the sumi lvalue. emitFlatFold's fold
        // tree is the gated scale_plus_min fold_model; the m_x*s_y mul is
        // recomputed INSIDE the fused expression, so the min brick's f32 result is
        // gate-only (never materialized standalone).
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY, mX, sY);
      } else if (loopBody.getFoldModel() == "scales_times_sumi") {
        // ---- q5_0 half-block asymmetric FIVE-BIT offset-binary packed-i4 (+ qh
        // 5th bit, `-16` bias) x i8 core + the ScalesTimesSumi fold. This is the
        // q4_1 op-by-op emit with FIVE deltas: (1) the integer core routes to
        // emitFiveBitOffsetBinaryDecodeProductValue (unsigned-nibble decode +
        // per-lane qh 5th-bit merge + `-16` offset-binary bias), (2) the 5th-bit
        // SOURCE is a per-block qh brick (re-read as two aligned u16 halves off the
        // SHARED weight base) sitting in the min-brick's slot -- NO m_x/s_y reads,
        // (3) the weight and activation quant offsets DIVERGE (weight qs@6 vs
        // activation qs@2/18, stamped by the front door), (4) the fold is the
        // ScalesTimesSumi tree `sumf + (d_x*d_y)*(float)sumi` with NO min term, and
        // (5) applyOffsetBias is true. Byte-identical to the monolithic q5_0
        // (five_bit_offset_binary / half-block / ScalesTimesSumi, m1, elided, mbf
        // 1) instance; the decode + fold + qh source are selected from the WALKED
        // op identity, not the fold_model string. ----
        tcrvrvv::FiveBitOffsetBinaryXI8ProductOp fiveBitProduct;
        loopBody.getBody().walk(
            [&](tcrvrvv::FiveBitOffsetBinaryXI8ProductOp o) {
              fiveBitProduct = o;
            });
        tcrvrvv::BlockFiveBitQhSourceOp qhBrick;
        loopBody.getBody().walk(
            [&](tcrvrvv::BlockFiveBitQhSourceOp o) { qhBrick = o; });
        if (!fiveBitProduct || !qhBrick || coreLoads.size() != 3 || !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q5_0 flat block-dot body requires the region integer core: "
              "three per-block loads (a u8 packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric five-bit offset-binary packed-i4 x i8 "
              "product, a lane0 scalar extract, and the per-block qh-source brick");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");
        // q5_0's ScalesTimesSumi fold has NO min correction: brick 2 must NOT carry
        // a min_term (fail-closed against a mis-stamped q4_1/q5_1 body).
        if (brick2.getMinTerm())
          return rewriter.notifyMatchFailure(
              brick2, "q5_0 brick 2 must NOT carry a min_term (the "
                      "ScalesTimesSumi fold has no per-block MIN correction)");
        // The KEY five-bit link (I7 fail-closed): the product's qh_source operand
        // must be the per-block qh brick's gate-only token. Deleting or
        // misdirecting the qh brick fails closed here.
        if (fiveBitProduct.getQhSource() != qhBrick.getResult())
          return rewriter.notifyMatchFailure(
              fiveBitProduct, "q5_0 five-bit product qh_source must be the "
                              "per-block qh-source brick's gate-only token");

        // Follow the five-bit product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight (u8), q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh.
        auto weightLoad =
            fiveBitProduct.getWeight().getDefiningOp<tcrvrvv::LoadOp>();
        auto lowLoad =
            fiveBitProduct.getActivationLow().getDefiningOp<tcrvrvv::LoadOp>();
        auto highLoad =
            fiveBitProduct.getActivationHigh().getDefiningOp<tcrvrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride())
          return rewriter.notifyMatchFailure(
              loopBody,
              "q5_0 five-bit product operands must be the region's per-block "
              "loads: a u8 packed-i4 weight load off the weight ABI buffer + two "
              "plain-i8 q8 (low/high) activation loads off the activation ABI "
              "buffer, each carrying a block_stride + quant_byte_offset");

        if (!coreReduce ||
            coreReduce.getInput() != fiveBitProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q5_0 integer-core reduce input must be the five-bit product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q5_0 integer-core lane0 extract input must be the "
                           "reduce");

        // The ScalesTimesSumi fold descriptor. Only descriptor.foldModel + the
        // shared state are consumed by emitFlatFold; the integer core + the qh
        // harvest are emitted op-by-op below, NOT via emitFlatBlockCore.
        FlatBlockDotDescriptor descriptor;
        descriptor.decodePrimitive = FlatDecodePrimitive::FiveBitOffsetBinary;
        descriptor.foldModel = FlatFoldModel::ScalesTimesSumi;
        descriptor.defaultCoreLmul = "m1";
        descriptor.applyOffsetBias = true;
        BlockDotFacts facts = deriveBlockDotFacts(loopBody, "m1");
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // Shared per-block base memo (byte-exact to the monolith's single
        // blockBaseValue): brick 1's + the qh brick's reads and the per-block loads
        // that name the same (%buffer, %block_index) SSA pair share ONE
        // `base + ib*stride`.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1's two per-block fp16 -> f32 SCALE reads (d_x/d_y at offset 0),
        // off the SHARED base (memo hits, no re-emit), byte-exact to fp16ReadAt.
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());

        // The qh field's TWO aligned 16-bit halves, read RIGHT AFTER brick 1's
        // dX/dY and BEFORE the sumi decl (byte-exact to the monolith's
        // dX,dY,qhLow16,qhHigh16 order at emitFlatBlockCore:5987-6004). Off the
        // SHARED weight base (memo hit) at the qh brick's OWN qh_byte_offset --
        // mutating the brick's qh_base operand or qh_byte_offset attr changes these
        // emitted addresses -> changes the bytes (anti-bypass); the descriptor qh
        // offset is NEVER read in this path. NOT fp16ReadAt: the qh read is a raw
        // `(uint16_t)*(const uint16_t *)` call to a u32 (no fcvt hint).
        mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
        llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
        mlir::Value qhBase = blockBaseFor(
            qhBrick.getQhBase(), qhBrick.getBlockIndex(),
            static_cast<int64_t>(qhBrick.getBlockStride().value_or(0)),
            "block_base_x");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_field"));
        auto u16ReadAt = [&](int64_t byteOffset) -> mlir::Value {
          mlir::Value ptr = rewriter.create<emitc::AddOp>(
              loc, qhBase.getType(), qhBase, sizeLit(byteOffset));
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                           u16ReadCallee, mlir::ValueRange{ptr})
              .getResult(0);
        };
        int64_t qhOffset =
            static_cast<int64_t>(qhBrick.getQhByteOffset().value_or(0));
        mlir::Value qhLow16 = u16ReadAt(qhOffset);
        mlir::Value qhHigh16 = u16ReadAt(qhOffset + 2);

        // The i32 sumi lvalue.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2). coreLmul is the
        // LOAD result LMUL (u8m1 for the weight; the vsetvl spelling is m1).
        auto loadVecType =
            llvm::cast<tcrvrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(qk / 2)};
            });

        // HETEROGENEOUS loads: the WEIGHT strip is loaded U8 (const uint8_t* /
        // __riscv_vle8_v_u8m1) while the two q8 activation halves stay I8. The
        // weight/activation quant offsets DIVERGE (weight@6, acts@2/18) -- the
        // front door stamped them; this emit reads them straight off the load ops.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type u8CoreType =
            emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        mlir::Type u8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTailI8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        auto emitLoadTailU8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTailU8(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTailI8(yb, static_cast<int64_t>(*lowLoad.getQuantByteOffset()));
        valueMap[highLoad.getLoaded()] = emitLoadTailI8(
            yb, static_cast<int64_t>(*highLoad.getQuantByteOffset()));

        // The five-bit offset-binary nibble+qh decode + asymmetric widening
        // product, emitted from the product op's OWN weight/low/high operands (via
        // the valueMap) + the re-read qh halves + chunkOffset 0 (the elided
        // single-strip base) -- the SAME emitFiveBitOffsetBinaryDecodeProductValue
        // arithmetic the monolith's FiveBitOffsetBinary strip reduce runs, so
        // byte-identical while op-sourced. applyOffsetBias=true is the q5_0 `-16`
        // vsub (q5_1 shares the fn with false).
        auto prodVecType = llvm::cast<tcrvrvv::VectorType>(
            fiveBitProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::Type u16WideType =
            emitc::OpaqueType::get(ctx, ("vuint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitFiveBitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(fiveBitProduct.getWeight()),
                valueMap.lookup(fiveBitProduct.getActivationLow()),
                valueMap.lookup(fiveBitProduct.getActivationHigh()), qhLow16,
                qhHigh16, chunkOffset, vl, i8CoreType, u8CoreType, u16WideType,
                i16WideType, coreLmul, wideLmul, 16, wideLmul, "i16", opName,
                role, /*applyOffsetBias=*/true);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[fiveBitProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand: seed a FRESH
        // literal-0 lane (per-block, no sumi carry), vwredsum the product, pull
        // lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // brick 1 ((d_x*d_y)*(float)sumi, ScalesTimesSumi) + brick 3 (sumf + term)
        // fold COLLECTIVELY into the one fused emitc.expression, fed the
        // operand-derived d_x/d_y + the sumi lvalue. NO min term (q5_0's fold has
        // no per-block MIN correction).
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY,
                     /*mX=*/mlir::Value(), /*sY=*/mlir::Value());
      } else {
        return rewriter.notifyMatchFailure(
            loopBody, "step-5b full body lowers the sumi_times_scales (q8_0), "
                      "left_assoc (q4_0), scale_plus_min (q4_1 min-only / q5_1 "
                      "five-bit+min), and scales_times_sumi (q5_0) folds; the "
                      "other flat fold trees are later steps");
      }
    } else {
      // ---- Step 1-3 SKELETON body (brick 3 stub, or brick 1 + brick 3): the
      // per-op brick lowering. The carried-IN acc maps to a LOAD of the sumf
      // lvalue at the TOP of the loop body; the carried-OUT acc_next maps to an
      // emitc.assign back into it at the BOTTOM (emitFlatBlockDot:5757,:5829).
      valueMap[coreBlock.getArgument(0)] = blockLoop.getInductionVar();
      valueMap[coreBlock.getArgument(1)] =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      for (mlir::Operation &op : coreBlock) {
        if (auto scaleProduct =
                llvm::dyn_cast<tcrvrvv::BlockFp16ScaleProductOp>(op)) {
          if (mlir::failed(emitBlockFp16ScaleProduct(
                  rewriter, loc, scaleProduct, valueMap,
                  /*bodyVL=*/mlir::Value())))
            return mlir::failure();
        } else if (auto accumulate =
                       llvm::dyn_cast<tcrvrvv::CrossBlockF32AccumulateOp>(op)) {
          if (mlir::failed(emitCrossBlockF32Accumulate(
                  rewriter, loc, accumulate, valueMap,
                  /*bodyVL=*/mlir::Value())))
            return mlir::failure();
        } else if (auto yield =
                       llvm::dyn_cast<tcrvrvv::TypedFlatBlockDotLoopYieldOp>(
                           op)) {
          mlir::Value accNext = valueMap.lookup(yield.getAccNext());
          if (!accNext)
            return rewriter.notifyMatchFailure(yield,
                                               "loop yield acc_next unmapped");
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumf", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumfVar, accNext);
        } else {
          return rewriter.notifyMatchFailure(
              &op,
              "skeleton loop body lowers the per-block scale product (brick 1) "
              "and the cross-block fold (brick 3); a full body must carry brick "
              "2 (the computed-scale dequant) so the region-driven full-chain "
              "path takes over");
        }
      }
    }
  }
  }

  // *s = sumf;  -- byte-exact to emitFlatBlockDot:5931-5945 (structured scalar
  // store through the output pointer).
  auto outPointer = llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
  if (!outPointer)
    return rewriter.notifyMatchFailure(loopBody,
                                       "loop-body output not a pointer");
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "store_s"));
  mlir::Value outIndex =
      rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
  emitc::SubscriptOp outSubscript =
      rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
  mlir::Value sumfFinal =
      rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
  rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitQ1_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ10Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ10Q80Op>(op))
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

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The binary sign decode runs ONE 32-lane sub-block body at the whole-LMUL
    // anchor whose i8 VLMAX spans the 32-element sub-block (m2 at VLEN128 where
    // e8m1 VLMAX is 16 < 32; m1 at VLEN256 where e8m1 VLMAX is 32). The gearbox
    // stamps integer_core_lmul from getRVVStripVLMAXElements; the default is m2 --
    // the VLEN-universal-safe floor (e8m2 VLMAX 32 spans the 32-element sub-block
    // at VLEN128 AND VLEN256), so an attr-less op lowers correctly at any VLEN; the
    // gearbox refines m2->m1 only at VLEN>=256. The 4 packed bit-bytes load
    // DIRECTLY into the i8 sign mask (vlm_v_b{ratio}, the packed bits ARE the mask:
    // bit 8b+i -> lane 8b+i), the i8 q8 quants are negated/merged in the i8 domain,
    // and ONE vwredsum widens i8->i16m1 per sub-block (no separate vwcvt; this is
    // ggml's shipped _vl128 lane structure).
    llvm::StringRef coreLmul = "m2";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    // The vbool ratio is SEW8/LMUL: m1 -> vbool8_t (vlm_v_b8), m2 -> vbool4_t
    // (vlm_v_b4). Derived from the anchor so the mask width tracks the LMUL flip.
    llvm::StringRef boolRatio = (coreLmul == "m2") ? "4" : "8";
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type maskType =
        emitc::OpaqueType::get(ctx, ("vbool" + boolRatio + "_t").str());

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                       // 128
    int64_t weightStride = blockDot.getWeightBlockStride(); // 18
    int64_t activationStride =
        blockDot.getActivationBlockStride();             // 34
    int64_t q8PerWeight = blockDot.getActivationBlocksPerWeight(); // 4
    int64_t weightQuantOffset = blockDot.getWeightQuantByteOffset(); // 2
    int64_t activationQuantOffset =
        blockDot.getActivationQuantByteOffset();         // 2
    int64_t subBlockElems = qk / q8PerWeight;            // 32 (q8 block lanes)
    int64_t bytesPerSubBlock = subBlockElems / 8;        // 4 bit bytes per q8 block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK1_0;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // The ONE sanctioned opaque scalar fp16->fp32 read (a typed emitc.call_opaque
    // node, exactly how the q4_0 sibling emits its fp16 scale reads).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1) { ... }.
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // const uint8_t *xb = vx + ib*18;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "block_base_x"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(weightStride));
      mlir::Value xb = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                     weightBase, xOff);
      // float d0 = (float)*(const _Float16 *)(xb);
      mlir::Value d0 = fp16Read(xb);

      // float sumi = 0.0f;  (RESET each super-block; the q8-sub-block fp32 fold)
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(floatType),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar,
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

      // The FOUR q8 sub-blocks are UNROLLED (each carries its own d1_k + 32-bit
      // sign decode), folded in strict ascending k order (fp non-associativity).
      for (int64_t k = 0; k < q8PerWeight; ++k) {
        // const uint8_t *yb = vy + (ib*4 + k)*34;
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "block_base_y"));
        mlir::Value ibTimes =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(q8PerWeight));
        mlir::Value yIdx = rewriter.create<emitc::AddOp>(loc, sizeType, ibTimes,
                                                         sizeLit(k));
        mlir::Value yOff = rewriter.create<emitc::MulOp>(loc, sizeType, yIdx,
                                                         sizeLit(activationStride));
        mlir::Value yb = rewriter.create<emitc::AddOp>(loc, activationPtrType,
                                                       activationBase, yOff);
        // float d1 = (float)*(const _Float16 *)(yb);
        mlir::Value d1 = fp16Read(yb);

        // The q8 sub-block is ONE 32-lane body (ggml's shipped _vl128 lane
        // structure): vlm_v_b{ratio} the 4 packed bit-bytes straight into the i8
        // sign mask, vle8 the 32 q8 quants, i8-domain vneg/vmerge -> signed q8,
        // ONE vwredsum widening i8 -> i16m1. No 8-lane sub-grouping, no kmask
        // table, no separate vwcvt: the integer dot is a single 32-lane reduce.

        // size_t vl = __riscv_vsetvl_e8{coreLmul}(32);  (the whole 32-element
        // sub-block; the anchor's i8 VLMAX spans it -- m2 at VLEN128, m1 at
        // VLEN256 -- so vl stays 32 and never crosses into a second sub-block).
        std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, coreLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, setvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(subBlockElems)};
            });

        // const uint8_t *qsbits = xb + 2 + k*4;  (the 4 packed bit-bytes = 32
        // sign bits of this sub-block).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bits_byte_addr"));
        int64_t bitByteOffset = weightQuantOffset + k * bytesPerSubBlock;
        mlir::Value bitsPtr = rewriter
                                  .create<emitc::AddOp>(loc, weightPtrType, xb,
                                                        sizeLit(bitByteOffset))
                                  .getResult();
        mlir::Value bitsPtrU8 =
            rewriter.create<emitc::CastOp>(loc, u8PtrType, bitsPtr).getResult();

        // vbool{ratio}_t is_not_zero = vlm_v_b{ratio}(qsbits, 32);  (the packed
        // bits ARE the i8 sign mask: bit 8b+i -> lane 8b+i, set -> +q8).
        std::string lmCallee = ("__riscv_vlm_v_b" + boolRatio).str();
        mlir::Value signMask =
            emitOpaqueCall(rewriter, loc, maskType, lmCallee,
                           mlir::ValueRange{bitsPtrU8, vl}, opName, role);

        // vint8{coreLmul}_t q8 = vle8(yb + 2);  (the 32 q8 quants of this block).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "q8_block_addr"));
        mlir::Value q8Ptr = rewriter
                                .create<emitc::AddOp>(loc, activationPtrType, yb,
                                                      sizeLit(activationQuantOffset))
                                .getResult();
        mlir::Value q8PtrI8 =
            rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Ptr).getResult();
        std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        mlir::Value q8 =
            emitOpaqueCall(rewriter, loc, i8CoreType, loadCallee,
                           mlir::ValueRange{q8PtrI8, vl}, opName, role);

        // sy = vmerge(vneg_i8(q8), q8, is_not_zero);  (i8-domain negate/merge,
        // ggml's exact ops: +q8 where bit set, -q8 where clear). The q8 quant
        // domain is [-127,127] (the -128 boundary never occurs in a real q8_0
        // quantization), so the i8 vneg is exact on every gate input.
        std::string negCallee = ("__riscv_vneg_v_i8" + coreLmul).str();
        mlir::Value q8Neg =
            emitOpaqueCall(rewriter, loc, i8CoreType, negCallee,
                           mlir::ValueRange{q8, vl}, opName, role);
        std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
        mlir::Value signedQ8 =
            emitOpaqueCall(rewriter, loc, i8CoreType, mergeCallee,
                           mlir::ValueRange{q8Neg, q8, signMask, vl}, opName,
                           role);

        // int sumi_block = vmv_x_s(vwredsum_i8{coreLmul}_i16m1(sy, 0, 32));  (ONE
        // widening reduce over the 32 lanes: i8 product chain summed into i16m1.
        // 32 lanes * |q8|<=127 = 4064 < 32767, so the i16 accumulator never
        // overflows.)
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 16, "m1", "i16");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i16m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroI16 = rewriter.create<emitc::LiteralOp>(
                  loc, emitc::OpaqueType::get(ctx, "int16_t"), "0");
              return {zeroI16, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i8" + coreLmul + "_i16m1").str();
        mlir::Value red =
            emitOpaqueCall(rewriter, loc, i16m1Type, reduceCallee,
                           mlir::ValueRange{signedQ8, seed, vl}, opName, role);
        std::string extractCallee = "__riscv_vmv_x_s_i16m1_i16";
        mlir::Type i16Type = emitc::OpaqueType::get(ctx, "int16_t");
        mlir::Value sumiBlockI16 =
            emitOpaqueCall(rewriter, loc, i16Type, extractCallee,
                           mlir::ValueRange{red}, opName, role);
        // int32_t sumi_block = (int)..;  (the sub-block integer dot result)
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi_block", opName, role));
        auto sumiBlockVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        mlir::Value sumiBlockI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, sumiBlockI16)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiBlockVar, sumiBlockI32);

        // sumi = sumi + d1 * (float)sumi_block;  (ggml EXACT order, grouped into
        // ONE emitc.expression so mlir-translate renders ONE C statement and the
        // compiler fuses the same FMA ggml does under -ffp-contract=on/default).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "fp32_accumulate_sub"));
        mlir::Value sumiBlockFinal =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiBlockVar)
                .getResult();
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, floatType, sumiVar).getResult();
        auto subExpr = rewriter.create<emitc::ExpressionOp>(
            loc, floatType, /*do_not_inline=*/false);
        {
          mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
          mlir::Block *exprBlock = rewriter.createBlock(&subExpr.getRegion());
          rewriter.setInsertionPointToStart(exprBlock);
          mlir::Value sumiBlockFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiBlockFinal)
                  .getResult();
          mlir::Value term =
              rewriter.create<emitc::MulOp>(loc, floatType, d1, sumiBlockFloat);
          mlir::Value sumiNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumiCur, term);
          rewriter.create<emitc::YieldOp>(loc, sumiNext);
        }
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, subExpr.getResult());
      }

      // sumf = sumf + d0 * sumi;  (the super-block term, ggml EXACT order, ONE
      // emitc.expression -> ONE C FMA).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiSuper =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d0, sumiSuper);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

// The typed per-block dual-fp16 SCALE reconstruction primitive. This lowers the
// SAME two scalar fp16->fp32 reads + scalar float multiply the monolithic
// block-dot emitters produce inline for `d_x * d_y`, but as a first-class typed
// body op (I5) rather than an opaque inline lambda. The emitted C is byte-EQUAL
// to the monolith's inline `d_x * d_y` because kFp16ScaleReadCallee currently
// holds the same spelling as the monolith's independent local fp16ReadCallee
// literals (:187, :479, :5428, :6022) and the float emitc.mul spelling matches
// -- a byte-equal-literal coincidence, NOT a mechanized single-source share
// (those monolith literals do not reference this constant). The consolidation
// that actually mechanizes drift-protection is deferred to brick(5) (the q8_0
// wire-in), where the monolith inline reads are replaced by this typed op. The
// op is scalar (no vl); bodyVL is unused.
mlir::LogicalResult VariantToEmitCFunc::emitBlockFp16ScaleProduct(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::BlockFp16ScaleProductOp scaleProduct,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type sizeType = getSizeType(rewriter);

  mlir::Value lhsBase = valueMap.lookup(scaleProduct.getLhsScaleBase());
  mlir::Value rhsBase = valueMap.lookup(scaleProduct.getRhsScaleBase());
  if (!lhsBase || !rhsBase)
    return rewriter.notifyMatchFailure(scaleProduct,
                                       "block_fp16_scale_product operand "
                                       "unmapped");

  llvm::StringRef opName = scaleProduct.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef role = scaleProduct.getTCRVEmitCLowerableSourceRole();

  // M-FLAT step 3 -- the loop-capable per-block-source form. When block_index
  // is present it is the enclosing tcrv_rvv.typed_flat_block_dot_loop_body
  // region's induction variable, so each per-block fp16 scale header lives at
  // `base + block_index*stride (+ byte_offset)`. This branch replicates
  // emitFlatBlockDot's blockBaseValue (:5451-5462, blockOffset 0) + fp16ReadAt
  // (:5467-5476) byte-exact: the SAME size_t emitc.mul + pointer emitc.add
  // block-base arithmetic and the SAME `(float)*(const _Float16 *)` call_opaque
  // read (spelled with fp16ReadAt's "fcvt.s.h" verbatim -- unlike the single-
  // block form below, which keeps its own frozen "(float)*(const _Float16 *)"
  // verbatim; only the per-block form is the byte-exact monolith replica). The
  // imported ABI bases stay the loop-invariant block-0 pointers; only the loop
  // offset is added. The single-block (block_index absent) path below is
  // untouched.
  if (mlir::Value blockIndex = scaleProduct.getBlockIndex()) {
    mlir::Value ib = valueMap.lookup(blockIndex);
    if (!ib)
      return rewriter.notifyMatchFailure(
          scaleProduct, "block_fp16_scale_product block_index unmapped");

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType,
                                               std::to_string(v));
    };
    // const uint8_t *xb = base + ib*stride;  -- byte-exact to blockBaseValue
    // (blockOffset 0): a size_t emitc.mul then a pointer emitc.add.
    auto perBlockBase = [&](mlir::Value base, int64_t stride,
                            const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, base.getType(), base, off);
    };
    // (float)*(const _Float16 *)(blockBase (+ byte_offset))  -- byte-exact to
    // fp16ReadAt: the optional pointer emitc.add for the header byte offset then
    // the sanctioned opaque read.
    auto perBlockRead = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset && *byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(), blockBase,
                                             sizeLit(*byteOffset));
      return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                            mlir::ValueRange{addr}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    mlir::Value xb = perBlockBase(
        lhsBase, static_cast<int64_t>(*scaleProduct.getLhsBlockStride()),
        "block_base_x");
    mlir::Value yb = perBlockBase(
        rhsBase, static_cast<int64_t>(*scaleProduct.getRhsBlockStride()),
        "block_base_y");
    mlir::Value dX = perBlockRead(xb, scaleProduct.getLhsScaleByteOffset());
    mlir::Value dY = perBlockRead(yb, scaleProduct.getRhsScaleByteOffset());
    // float scale = d_x * d_y;  (ggml's q8_0 scale order: scales multiplied
    // FIRST)
    mlir::Value scale =
        rewriter.create<emitc::MulOp>(loc, floatType, dX, dY).getResult();
    valueMap[scaleProduct.getResult()] = scale;
    return mlir::success();
  }

  // Per-block fp16 read at `base (+ byte_offset)`. Default offset 0 reads the
  // AoS fp16 header at the block base (matching the monolithic q8_0 read).
  auto fp16Read = [&](mlir::Value base,
                      std::optional<int64_t> byteOffset) -> mlir::Value {
    mlir::Value readBase = base;
    if (byteOffset && *byteOffset != 0) {
      mlir::Value offset =
          rewriter.create<emitc::LiteralOp>(loc, sizeType,
                                            std::to_string(*byteOffset));
      readBase = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                               offset);
    }
    return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                          mlir::ValueRange{readBase}, opName, role);
  };

  mlir::Value dX = fp16Read(lhsBase, scaleProduct.getLhsScaleByteOffset());
  mlir::Value dY = fp16Read(rhsBase, scaleProduct.getRhsScaleByteOffset());
  // float scale = d_x * d_y;  (ggml's q8_0 scale order: scales multiplied FIRST)
  mlir::Value scale =
      rewriter.create<emitc::MulOp>(loc, floatType, dX, dY).getResult();
  valueMap[scaleProduct.getResult()] = scale;
  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitBlockComputedScaleDequant(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::BlockComputedScaleDequantOp dequant,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");

  mlir::Value sumi = valueMap.lookup(dequant.getSumi());
  mlir::Value scale = valueMap.lookup(dequant.getComputedScale());
  if (!sumi || !scale)
    return rewriter.notifyMatchFailure(dequant,
                                       "block_computed_scale_dequant operand "
                                       "unmapped");

  // float term = (float)sumi * scale;  -- byte-identical at the operation
  // spelling level to the monolithic block-dot fold: the SAME i32 -> float
  // emitc.cast (the `(float)sumi` sitofp) + the SAME scalar float emitc.mul the
  // monolith produces inline for the per-block `(float)sumi * <scale>` term.
  // scale is the COMPUTED tcrv_rvv.block_fp16_scale_product output (d_x*d_y),
  // not an imported ABI scale. This op stops at the per-block term; the
  // cross-block fp32 accumulate (`sumf += term`) is a separate typed step.
  mlir::Value sumiFloat =
      rewriter.create<emitc::CastOp>(loc, floatType, sumi).getResult();
  mlir::Value term =
      rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, scale)
          .getResult();
  valueMap[dequant.getResult()] = term;
  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitCrossBlockF32Accumulate(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::CrossBlockF32AccumulateOp accumulate,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");

  mlir::Value acc = valueMap.lookup(accumulate.getAcc());
  mlir::Value term = valueMap.lookup(accumulate.getTerm());
  if (!acc || !term)
    return rewriter.notifyMatchFailure(accumulate,
                                       "cross_block_f32_accumulate operand "
                                       "unmapped");

  // float sumf = acc + term;  -- byte-identical at the operation-spelling level
  // to the monolithic block-dot cross-block fold: the SAME scalar float
  // emitc.add the monolith produces for `sumf + <block term>`. The caller folds
  // in STRICT ascending block order (block-carried), so the fp non-associativity
  // matches ggml byte-for-byte. acc is the block-carried f32 accumulator, term
  // is the COMPUTED tcrv_rvv.block_computed_scale_dequant output
  // (`(float)sumi * scale`). The op stops at the fold; the block loop and the
  // final scalar store are separate typed steps.
  mlir::Value sumfNext =
      rewriter.create<emitc::AddOp>(loc, floatType, acc, term).getResult();
  valueMap[accumulate.getResult()] = sumfNext;
  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitTypedVectorLane0ToScalarExtract(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::TypedVectorLane0ToScalarExtractOp extract,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");

  mlir::Value input = valueMap.lookup(extract.getInput());
  if (!input)
    return rewriter.notifyMatchFailure(
        extract, "typed_vector_lane0_to_scalar_extract input unmapped");

  // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(red);  -- byte-identical at the
  // operation-spelling level to the monolithic block-dot lane0 extraction
  // (emitFlatBlockDot:5653-5655): the SAME __riscv_vmv_x_s_i32m1_i32 call_opaque
  // that pulls the vwredsum lane0 into the scalar sumi. The intrinsic targets
  // lane 0 regardless of vl, so the call takes ONLY the i32m1 vector value (the
  // op's vl operand is the boundary marker, not a call argument).
  std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
  mlir::Value scalar =
      emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                     mlir::ValueRange{input},
                     extract.getTCRVEmitCLowerableSourceOpName(),
                     extract.getTCRVEmitCLowerableSourceRole());
  valueMap[extract.getResult()] = scalar;
  return mlir::success();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
