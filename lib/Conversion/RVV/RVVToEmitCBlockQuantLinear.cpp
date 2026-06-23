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

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-block reduction anchor). It is the
    // *how* (vector grouping / strip width), never the *what*: the dot product is
    // byte-exact either way (vwredsum sums the same integer set; integer add is
    // order-independent). The chosen i8 source LMUL ("mf4"/"m1") drives the
    // widened i16 product LMUL ("mf2"/"m2") and the i8 load/vsetvl spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    // The two outer-loop SHAPE knobs (also bounded resource/scheduling facts, the
    // *how* not the *what*; both are byte-exact -- the fp32 folds stay in strict
    // ascending block order, only the independent integer cores overlap). The
    // verifier bounds the factor to 1|2|4 and the elision to robust|elided, and
    // forbids "elided" unless the integer core anchors at m1.
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (the verifier bounds the
    // source to mf4|m1, so the product is mf2|m2 respectively).
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    // The i8<core> / i16<wide> / i32m1 vector types of the integer core (same
    // spelling convertVectorTypeToEmitC produces; the inner work is INC-1's
    // decode/product chain, re-anchored at the chosen LMUL).
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // Per-block address arithmetic for a block at outer-loop value ib plus a
    // compile-time block offset within the unroll group: const uint8_t *xb =
    // vx + (ib + blockOffset)*18; const uint8_t *yb = vy + (ib + blockOffset)*34.
    // For the no-unroll (factor 1) form blockOffset is 0 and the resulting nodes
    // (mul ib*18, add base, mul ib*34, add base) are byte-identical to the
    // committed single-block emission.
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

    // The two scalar fp16->fp32 reads (the ONE sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // The offset-binary asymmetric i4xi8 decode/product + reduce for ONE strip,
    // seeded with the carried sumi, returning the next scalar sumi (INC-1 nodes
    // re-anchored at the chosen integer-core LMUL). vl is the active strip VL;
    // chunkOffset is the within-half-block byte offset (a value: the strip
    // induction var for the robust loop, or a 0 literal for the elided core).
    // When carrySumi is set the seed lane0 is loaded from sumiVar (the robust
    // strip carries the running sumi across strips); otherwise the seed lane0 is
    // a "0" literal (the elided core runs the strip exactly once). The seed load
    // is emitted at the seed position (right before vmv_v_x), preserving the
    // committed single-block node order byte-for-byte.
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      };
      std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadChunk = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         loadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      mlir::Value w = loadChunk(chunkPtr(xb, weightPtrType, quantOffset));
      mlir::Value y0 = loadChunk(chunkPtr(yb, activationPtrType, quantOffset));
      mlir::Value y1 =
          loadChunk(chunkPtr(yb, activationPtrType, quantOffset + highOffset));

      mlir::FailureOr<mlir::Value> product = emitOffsetBinaryDecodeProductValue(
          rewriter, loc, w, y0, y1, vl, i8CoreType, i16WideType, "i8", coreLmul,
          16, wideLmul, "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      // vwredsum is LMUL-independent in result: the i16<wide> source reduces
      // into i32m1 lane 0 (the same integer set; the callee tracks the source
      // LMUL).
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce, and returns the sumi lvalue (the caller folds it in order). When
    // forceRobust is set the inner strip loop is always kept (the nb % factor
    // tail and the factor-1/robust shapes); otherwise stripElided selects the
    // single-vsetvl(16) elided core (m1-only, VLEN >= 128).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      // int32_t sumi = 0;  (RESET each block -- the integer dot is per-block)
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) (caps the
        // active vl at 16 when VLMAX >= 16, covering the whole half-block) + ONE
        // strip reduce. NO inner strip loop, NO sumi carry (seed lane0 = 0; the
        // strip can only run once).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, sizeLit(0), vl, sumiVar, /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 16 weight bytes, the loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). The mf4 core uses vsetvl_e32m1 (4 strips at
      // VLEN=128, INC-2a byte-identical); the m1 core uses vsetvl_e8m1 (ONE
      // strip at VLEN=128). Both stay VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        // size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, c, vl, sumiVar, /*carrySumi=*/true);
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
    };

    // The left-associative fp32 accumulate sumf = sumf + ((float)sumi*d_x)*d_y
    // (ggml EXACT order), grouped into ONE emitc.expression so mlir-translate
    // renders it as a SINGLE C statement and the compiler fuses the SAME FMA
    // ggml does under -ffp-contract=on/default (byte-exact across all four
    // modes). The two emitc.load temps stay OUTSIDE the expression (load lacks
    // the CExpression trait). For multi-block unrolls the caller invokes this in
    // STRICT ascending block order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value timesDx =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block (addresses + scales + integer core + fold) at outer-loop
    // value ib plus a compile-time block offset within the unroll group. The
    // integer cores of several blocks in a group may interleave, but emitFold is
    // invoked in strict ascending blockOffset order by the caller, so the fp32
    // folds stay byte-exact.
    // The integer core of ONE block (addresses + scales + sumi), WITHOUT the
    // fold. The factor independent cores of an unroll group are emitted
    // back-to-back so the two/four reductions sit adjacent and overlap (the
    // latency-overlap lever); the folds are emitted afterwards in strict
    // ascending block order. dX/dY/sumi are returned so the caller can fold in
    // order. For factor 1 the caller emits the core then its fold immediately,
    // so the node order is unchanged from the committed single-block emission.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      mlir::Value dX = fp16Read(xb);
      mlir::Value dY = fp16Read(yb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY};
    };

    if (multiBlockFactor == 1) {
      // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the no-unroll form;
      // byte-identical to the committed single-block emission (factor absent or
      // 1, robust or elided per strip_elision). The core then its fold emit
      // back-to-back, preserving the committed node order.
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY);
    } else {
      // Multi-block unroll: a main loop stepping by factor over the
      // nb - nb%factor full groups -- emit ALL factor independent integer cores
      // FIRST (so the factor reductions are adjacent and the compiler overlaps
      // them: the latency-overlap lever), THEN the factor folds in strict
      // ascending block order (the fp non-associativity boundary) -- then a
      // robust single-block scalar tail over the nb % factor remainder.
      //
      // size_t nb_main = nb - nb % factor;  (the largest multiple of factor)
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
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY);
      }
      // for (size_t ibt = nb_main; ibt < nb; ibt += 1) { robust single block }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
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

mlir::LogicalResult VariantToEmitCFunc::emitQ5_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
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

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-block reduction anchor). The chosen
    // i8 source LMUL drives the widened i16 product LMUL and the i8/u16 spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (mf4->mf2, m1->m2). The
    // u16 bit-extraction runs at the SAME wide LMUL.
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    std::string u16WideTypeName = ("vuint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type u16WideType = emitc::OpaqueType::get(ctx, u16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();          // weight qs (6)
    int64_t actQuantOffset = blockDot.getActivationQuantByteOffset(); // q8 qs (2)
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t qhOffset = blockDot.getWeightQhByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";
    // The qh 5th-bit field is read as TWO aligned 16-bit halves (NOT one 32-bit
    // load): the q5_0 stride is 22 and qhOffset is 2, so xb+2 and xb+4 are
    // 2-aligned for any block, whereas a uint32_t load at xb+2 would be MISALIGNED
    // (UB / trap-emulate). Little-endian, the low/high halves are exactly the two
    // uint16_t at qhOffset and qhOffset+2, value-identical to (qh & 0xFFFF) /
    // (qh >> 16) -- so the decode is unchanged, just alignment-safe (matching
    // _generic's memcpy safety, the byte-exact value is the same).

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*22;
    // const uint8_t *yb = vy + (ib+blockOffset)*34.
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

    // The scalar fp16->fp32 read (the sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };
    // The block_q5_0 per-element 5th-bit field, read as two ALIGNED 16-bit halves
    // (the qh bits for elements 0..15 at xb+qhOffset; for elements 16..31 at
    // xb+qhOffset+2). Each is a `(uint16_t)*(const uint16_t *)` scalar read (the
    // sanctioned opaque piece), 2-aligned and value-identical to the LE
    // (qh & 0xFFFF) / (qh >> 16) halves the decode broadcasts per strip.
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
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
      mlir::Value low16 = readHalf(qhOffset);
      mlir::Value high16 = readHalf(qhOffset + 2);
      return QhHalves{low16, high16};
    };

    // The 5-bit offset-binary asymmetric decode/product + reduce for ONE strip,
    // seeded with the carried sumi, returning the next scalar sumi.
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value qhLow16, mlir::Value qhHigh16,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed, mlir::Type castType) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
      };
      // The weight nibble bytes load as UNSIGNED (the decode masks/shifts run on
      // the u8 lane); the q8 halves load as plain SIGNED i8.
      std::string wLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      std::string yLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadW = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, wLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                         wLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, yLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         yLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      // The q5_0 WEIGHT nibbles are at quantOffset (6); the q8_0 ACTIVATION int8
      // quants are at actQuantOffset (2, distinct -- the q8_0 block has no qh).
      mlir::Value w = loadW(chunkPtr(xb, weightPtrType, quantOffset, u8PtrType));
      mlir::Value y0 =
          loadY(chunkPtr(yb, activationPtrType, actQuantOffset, i8PtrType));
      mlir::Value y1 = loadY(chunkPtr(yb, activationPtrType,
                                      actQuantOffset + highOffset, i8PtrType));

      mlir::FailureOr<mlir::Value> product =
          emitFiveBitOffsetBinaryDecodeProductValue(
              rewriter, loc, w, y0, y1, qhLow16, qhHigh16, chunkOffset, vl,
              i8CoreType, u8CoreType, u16WideType, i16WideType, coreLmul,
              wideLmul, 16, wideLmul, "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce(s), and returns the sumi lvalue (the caller folds it in order).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb, mlir::Value qhLow16,
            mlir::Value qhHigh16,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) + ONE strip
        // reduce. NO inner strip loop, NO sumi carry (c = 0, seed lane0 = 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
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

      // Robust core: the inner strip loop over the 16 weight bytes. The loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

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
    };

    // The fp32 accumulate sumf = sumf + (d_x*d_y)*(float)sumi (ggml q5_0/q8_0
    // scales-first order), grouped into ONE emitc.expression so mlir-translate
    // renders it as a SINGLE C statement and the compiler fuses the SAME FMA
    // ggml does under -ffp-contract=on/default (byte-exact across all four
    // modes). The two emitc.load temps stay OUTSIDE the expression. For
    // multi-block unrolls the caller invokes this in STRICT ascending block
    // order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value scales =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scales, sumiFloat);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      mlir::Value dX = fp16Read(xb);
      mlir::Value dY = fp16Read(yb);
      QhHalves qh = qhRead(xb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, qh.low16, qh.high16, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY};
    };

    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY);
    } else {
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
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
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

mlir::LogicalResult VariantToEmitCFunc::emitQ5_1Q8_1BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
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

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-half-block anchor). The chosen i8
    // source LMUL drives the widened i16 product LMUL and the i8/u16 spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (mf4->mf2, m1->m2). The
    // u16 bit-extraction runs at the SAME wide LMUL.
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    std::string u16WideTypeName = ("vuint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type u16WideType = emitc::OpaqueType::get(ctx, u16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();          // weight qs (8)
    int64_t actQuantOffset = blockDot.getActivationQuantByteOffset(); // q8 qs (4)
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t qhOffset = blockDot.getWeightQhByteOffset();
    int64_t weightMinOffset = blockDot.getWeightMinByteOffset();
    int64_t activationSumOffset = blockDot.getActivationSumByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";
    // The qh 5th-bit field is read as TWO aligned 16-bit halves (NOT one 32-bit
    // load): the q5_1 stride is 24 and qhOffset is 4, so xb+4 and xb+6 are
    // 2-aligned for any block, whereas a uint32_t load at xb+4 is fine but the
    // two-u16 form matches the q5_0 emitter and stays alignment-safe. Little-
    // endian, the low/high halves are exactly (qh & 0xFFFF) / (qh >> 16).

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*24;
    // const uint8_t *yb = vy + (ib+blockOffset)*36.
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
    // piece, a typed emitc.call_opaque node). q5_1 reads FOUR per block: d_x at
    // weight+0, m_x at weight+`weight_min_byte_offset`; d_y at activation+0, s_y
    // at activation+`activation_sum_byte_offset`.
    auto fp16ReadAt = [&](mlir::Value blockBase, mlir::Type ptrType,
                          int64_t byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, ptrType, blockBase,
                                             sizeLit(byteOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee, mlir::ValueRange{addr})
          .getResult(0);
    };
    // The block_q5_1 per-element 5th-bit field, read as two ALIGNED 16-bit halves
    // (the qh bits for elements 0..15 at xb+qhOffset; for elements 16..31 at
    // xb+qhOffset+2). Each is a `(uint16_t)*(const uint16_t *)` scalar read, the
    // sanctioned opaque piece, value-identical to the LE (qh & 0xFFFF) /
    // (qh >> 16) halves the decode broadcasts per strip.
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
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
      mlir::Value low16 = readHalf(qhOffset);
      mlir::Value high16 = readHalf(qhOffset + 2);
      return QhHalves{low16, high16};
    };

    // The 5-bit UNSIGNED decode/product + reduce for ONE strip, seeded with the
    // carried sumi, returning the next scalar sumi. The decode SHARES q5_0's
    // emitter with applyOffsetBias=false so the `-16` is dropped (the q5_1 weight
    // is an unsigned q5 in [0,31]; the bias lives in the per-block MIN scale).
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value qhLow16, mlir::Value qhHigh16,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed, mlir::Type castType) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
      };
      // The weight nibble bytes load as UNSIGNED (the decode masks/shifts run on
      // the u8 lane); the q8 halves load as plain SIGNED i8.
      std::string wLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      std::string yLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadW = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, wLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                         wLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, yLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         yLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      // The q5_1 WEIGHT nibbles are at quantOffset (8); the q8_1 ACTIVATION int8
      // quants are at actQuantOffset (4, distinct -- the q8_1 block has no qh but
      // carries the precomputed sum s before its quants).
      mlir::Value w = loadW(chunkPtr(xb, weightPtrType, quantOffset, u8PtrType));
      mlir::Value y0 =
          loadY(chunkPtr(yb, activationPtrType, actQuantOffset, i8PtrType));
      mlir::Value y1 = loadY(chunkPtr(yb, activationPtrType,
                                      actQuantOffset + highOffset, i8PtrType));

      mlir::FailureOr<mlir::Value> product =
          emitFiveBitOffsetBinaryDecodeProductValue(
              rewriter, loc, w, y0, y1, qhLow16, qhHigh16, chunkOffset, vl,
              i8CoreType, u8CoreType, u16WideType, i16WideType, coreLmul,
              wideLmul, 16, wideLmul, "i16", opName, role,
              /*applyOffsetBias=*/false);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce(s), and returns the sumi lvalue (the caller folds it in order).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb, mlir::Value qhLow16,
            mlir::Value qhHigh16,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) + ONE strip
        // reduce. NO inner strip loop, NO sumi carry (c = 0, seed lane0 = 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
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

      // Robust core: the inner strip loop over the 16 weight bytes. The loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

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
    };

    // The fp32 accumulate sumf = sumf + (d_x*d_y)*sumi + m_x*s_y (ggml q5_1 EXACT
    // statement, quants.c:394 -- IDENTICAL to q4_1's), grouped into ONE
    // emitc.expression so mlir-translate renders it as a SINGLE C statement
    // matching ggml's fp associativity / contraction byte-for-byte. The two
    // summed products (the scale term and the MIN term) are the Family-B
    // distinction from q5_0's single block term. The emitc.load temps stay
    // OUTSIDE the expression. For multi-block unrolls the caller invokes this in
    // STRICT ascending block order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY,
                        mlir::Value mX, mlir::Value sY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // (d_x * d_y) FIRST, then * sumi -- ggml q5_1's scale term order.
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value scaleTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scaleProduct,
                                          sumiFloat);
        // m_x * s_y -- the per-block MIN/SUM correction term.
        mlir::Value minTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, mX, sY);
        // ggml's EXACT tree (quants.c:394): `sumf += (d_x*d_y)*sumi + m_x*s_y;`.
        // The `+` between the two products binds BEFORE the `+=`, so the two
        // products are summed FIRST, then added to sumf: sumf + (scaleTerm +
        // minTerm). This grouping is fp-significant (`(sumf+A)+B != sumf+(A+B)`)
        // -- byte-exactness requires this exact tree.
        mlir::Value blockTerm =
            rewriter.create<emitc::AddOp>(loc, floatType, scaleTerm, minTerm);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block's integer core (addresses + four scales + qh + sumi),
    // WITHOUT the fold; the caller folds in strict ascending block order.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
      mlir::Value mX;
      mlir::Value sY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      // ggml reads d_x then d_y for the scale product, then m_x and s_y for the
      // MIN term; read them in that order for a stable node sequence.
      mlir::Value dX = fp16ReadAt(xb, weightPtrType, 0);
      mlir::Value dY = fp16ReadAt(yb, activationPtrType, 0);
      mlir::Value mX = fp16ReadAt(xb, weightPtrType, weightMinOffset);
      mlir::Value sY = fp16ReadAt(yb, activationPtrType, activationSumOffset);
      QhHalves qh = qhRead(xb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, qh.low16, qh.high16, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY, mX, sY};
    };

    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
    } else {
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
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY, core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // A typed i8m1 chunk load: __riscv_vle8_v_i8m1(ptr + fixed, vl).
    std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    auto loadChunk = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                         mlir::Value vl) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                       loadCallee, mlir::ValueRange{cast, vl})
          .getResult(0);
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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, setvlCallee));
      mlir::Value vl =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);

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
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seedCallee));
          mlir::Value sumiSeed =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          mlir::Value seed =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i32m1Type}, seedCallee,
                      mlir::ValueRange{sumiSeed, sizeLit(1)})
                  .getResult(0);
          std::string reduceCallee =
              ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reduceCallee));
          mlir::Value red =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                      mlir::ValueRange{*product, seed, vl})
                  .getResult(0);
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extractCallee));
          mlir::Value sumi =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                               extractCallee,
                                               mlir::ValueRange{red})
                  .getResult(0);

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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // A typed i8m1 chunk load: __riscv_vle8_v_i8m1(ptr + fixed, vl).
    std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    auto loadChunk = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                         mlir::Value vl) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                       loadCallee, mlir::ValueRange{cast, vl})
          .getResult(0);
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
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, setvlCallee));
          mlir::Value vl =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{sizeType}, setvlCallee,
                      mlir::ValueRange{sizeLit(halfBlock)})
                  .getResult(0);

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
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, seedCallee));
              mlir::Value sumiSeed =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              mlir::Value seed =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{i32m1Type}, seedCallee,
                          mlir::ValueRange{sumiSeed, sizeLit(1)})
                      .getResult(0);
              std::string reduceCallee =
                  ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, reduceCallee));
              mlir::Value red =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                          mlir::ValueRange{*product, seed, vl})
                      .getResult(0);
              std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, extractCallee));
              mlir::Value sumi =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{i32Type}, extractCallee,
                          mlir::ValueRange{red})
                      .getResult(0);

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
      step(i8LoadCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       i8LoadCallee,
                                       mlir::ValueRange{cast, vl8})
          .getResult(0);
    };
    // The repacked nibbles already carry the ^0x88 offset-binary bias, so the
    // decode is a plain sign-extension: b_lo = vsra(vsll(b,4),4); b_hi=vsra(b,4).
    std::string sllCallee = ("__riscv_vsll_vx_i8" + l8).str();
    std::string sraCallee = ("__riscv_vsra_vx_i8" + l8).str();
    mlir::Value four = sizeLit(4);
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      step(sllCallee);
      mlir::Value shl =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                           sllCallee,
                                           mlir::ValueRange{packed, four, vl8})
              .getResult(0);
      step(sraCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       sraCallee,
                                       mlir::ValueRange{shl, four, vl8})
          .getResult(0);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      step(sraCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       sraCallee,
                                       mlir::ValueRange{packed, four, vl8})
          .getResult(0);
    };
    // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k]:
    // *(const int8_t *)(ab + 8 + k).
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      step("act_quant_scalar");
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       i8ReadCallee, mlir::ValueRange{cast})
          .getResult(0);
    };
    // vwmacc_vx widening multiply-accumulate: acc += scalar * vec (i8->i16).
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      step(vwmaccCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                       vwmaccCallee,
                                       mlir::ValueRange{acc, scalar, vec, vl8})
          .getResult(0);
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
            step(fmvCallee);
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                    .getResult();
            sumf[c] =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                                 fmvCallee,
                                                 mlir::ValueRange{zero, vl8})
                    .getResult(0);
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
              step(mvCallee);
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                               mvCallee,
                                               mlir::ValueRange{zero, vl8})
                  .getResult(0);
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
              step(vwaddCallee);
              sumi32[c] =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{i32m2Type},
                                                   vwaddCallee,
                                                   mlir::ValueRange{lo, hi, vl8})
                      .getResult(0);
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
            step(f16LoadCallee);
            mlir::Value bD =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f16m1Type},
                                                 f16LoadCallee,
                                                 mlir::ValueRange{dCast, vl8})
                    .getResult(0);

            // d_c = vfwmul_vf(b_d, *(const _Float16 *)&al.d[c], 8);  -- the raw
            // _Float16 activation scale (NO float cast).
            std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
            std::string vfcvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, l32,
                                                         "f32");
            std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
            llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
            for (int64_t c = cLo; c < cHi; ++c) {
              step("act_scale_scalar");
              mlir::Value aDOff = rewriter.create<emitc::AddOp>(
                  loc, activationPtrType, al, sizeLit(c * 2));
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, aDOff)
                      .getResult();
              mlir::Type f16ScalarType =
                  emitc::OpaqueType::get(ctx, "_Float16");
              mlir::Value aD =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{f16ScalarType}, f16ReadCallee,
                          mlir::ValueRange{aDCast})
                      .getResult(0);
              step(vfwmulCallee);
              mlir::Value dC =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{f32m2Type},
                                                   vfwmulCallee,
                                                   mlir::ValueRange{bD, aD, vl8})
                      .getResult(0);
              // sumf_c = vfmacc_vv(sumf_c, vfcvt_f_x_v(sumi_c, 8), d_c, 8);
              step(vfcvtCallee);
              mlir::Value sumiF =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{f32m2Type},
                                                   vfcvtCallee,
                                                   mlir::ValueRange{sumi32[c],
                                                                    vl8})
                      .getResult(0);
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                      .getResult();
              step(vfmaccCallee);
              mlir::Value nextF =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{f32m2Type}, vfmaccCallee,
                          mlir::ValueRange{curF, sumiF, dC, vl8})
                      .getResult(0);
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
            step(vseCallee);
            rewriter.create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{}, vseCallee,
                mlir::ValueRange{dst, sumfVal, vl8});
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
      step(i8LoadCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       i8LoadCallee,
                                       mlir::ValueRange{cast, vl8})
          .getResult(0);
    };
    // The repacked nibbles already carry the ^0x88 offset-binary bias, so the
    // decode is a plain sign-extension: b_lo = vsra(vsll(b,4),4); b_hi=vsra(b,4).
    std::string sllCallee = ("__riscv_vsll_vx_i8" + l8).str();
    std::string sraCallee = ("__riscv_vsra_vx_i8" + l8).str();
    mlir::Value four = sizeLit(4);
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      step(sllCallee);
      mlir::Value shl =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                           sllCallee,
                                           mlir::ValueRange{packed, four, vl8})
              .getResult(0);
      step(sraCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       sraCallee,
                                       mlir::ValueRange{shl, four, vl8})
          .getResult(0);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      step(sraCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       sraCallee,
                                       mlir::ValueRange{packed, four, vl8})
          .getResult(0);
    };
    // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k]:
    // *(const int8_t *)(ab + 2 + k).
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      step("act_quant_scalar");
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       i8ReadCallee, mlir::ValueRange{cast})
          .getResult(0);
    };
    // vwmacc_vx widening multiply-accumulate: acc += scalar * vec (i8->i16).
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      step(vwmaccCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                       vwmaccCallee,
                                       mlir::ValueRange{acc, scalar, vec, vl8})
          .getResult(0);
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
        step(fmvCallee);
        mlir::Value zero =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                .getResult();
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                         fmvCallee,
                                         mlir::ValueRange{zero, vl8})
            .getResult(0);
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
          step(mvCallee);
          mlir::Value zero =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                           mvCallee,
                                           mlir::ValueRange{zero, vl8})
              .getResult(0);
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
          step(vwaddCallee);
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m2Type},
                                           vwaddCallee,
                                           mlir::ValueRange{lo, hi, vl8})
              .getResult(0);
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
          step(f16LoadCallee);
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f16m1Type},
                                           f16LoadCallee,
                                           mlir::ValueRange{dCast, vl8})
              .getResult(0);
        };
        llvm::SmallVector<mlir::Value> bD;
        for (int64_t h = 0; h < numHalves; ++h)
          bD.push_back(loadScales(h * half));

        // The single activation scale *(const _Float16 *)&al.d (NO float cast),
        // broadcast into both halves' vfwmul.
        step("act_scale_scalar");
        mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
        llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
        mlir::Value aDCast =
            rewriter.create<emitc::CastOp>(loc, f16PtrType, al).getResult();
        mlir::Value aD =
            rewriter
                .create<emitc::CallOpaqueOp>(loc,
                                             mlir::TypeRange{f16ScalarType},
                                             f16ReadCallee,
                                             mlir::ValueRange{aDCast})
                .getResult(0);

        // d_{a,b} = vfwmul_vf(b_d_{a,b}, aD, 8);
        // sumf_{a,b} = vfmacc_vv(sumf_{a,b}, vfcvt_f_x_v(sumi_{a,b},8), d, 8);
        std::string vfwmulCallee = ("__riscv_vfwmul_vf_f32" + l32).str();
        std::string vfcvtCallee =
            riscvIntrinsicName("vfcvt_f_x_v", 32, l32, "f32");
        std::string vfmaccCallee = ("__riscv_vfmacc_vv_f32" + l32).str();
        auto fold = [&](mlir::Value bD, mlir::Value sumi,
                        mlir::Value sumfVar) {
          step(vfwmulCallee);
          mlir::Value dC =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfwmulCallee,
                                               mlir::ValueRange{bD, aD, vl8})
                  .getResult(0);
          step(vfcvtCallee);
          mlir::Value sumiF =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfcvtCallee,
                                               mlir::ValueRange{sumi, vl8})
                  .getResult(0);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar)
                  .getResult();
          step(vfmaccCallee);
          mlir::Value nextF =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfmaccCallee,
                                               mlir::ValueRange{curF, sumiF, dC,
                                                                vl8})
                  .getResult(0);
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
        step(vseCallee);
        rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, vseCallee,
                                             mlir::ValueRange{dst, sumfVal,
                                                              vl8});
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
      step(u8LoadCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8mf2Type},
                                       u8LoadCallee,
                                       mlir::ValueRange{cast, vl8})
          .getResult(0);
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
      step(reinterpretCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       reinterpretCallee, mlir::ValueRange{u})
          .getResult(0);
    };
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      step(vandCallee);
      mlir::Value mask =
          rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x0F").getResult();
      mlir::Value lo =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8mf2Type},
                                           vandCallee,
                                           mlir::ValueRange{packed, mask, vl8})
              .getResult(0);
      return reinterpretToI8(lo);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      step(vsrlCallee);
      mlir::Value four =
          rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x04").getResult();
      mlir::Value hi =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8mf2Type},
                                           vsrlCallee,
                                           mlir::ValueRange{packed, four, vl8})
              .getResult(0);
      return reinterpretToI8(hi);
    };
    // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k].
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      step("act_quant_scalar");
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       i8ReadCallee, mlir::ValueRange{cast})
          .getResult(0);
    };
    // vwmacc_vx widening multiply-accumulate: acc += scalar * vec (i8->i16).
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      step(vwmaccCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                       vwmaccCallee,
                                       mlir::ValueRange{acc, scalar, vec, vl8})
          .getResult(0);
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
        step(fmvCallee);
        mlir::Value zero =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                .getResult();
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                         fmvCallee,
                                         mlir::ValueRange{zero, vl8})
            .getResult(0);
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
          step(mvCallee);
          mlir::Value zero =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                           mvCallee,
                                           mlir::ValueRange{zero, vl8})
              .getResult(0);
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
          step(vwaddCallee);
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m2Type},
                                           vwaddCallee,
                                           mlir::ValueRange{lo, hi, vl8})
              .getResult(0);
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
          step(f16LoadCallee);
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f16m1Type},
                                           f16LoadCallee,
                                           mlir::ValueRange{dCast, vl8})
              .getResult(0);
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
        step("act_scale_scalar");
        mlir::Type f16ScalarType = emitc::OpaqueType::get(ctx, "_Float16");
        llvm::StringRef f16ReadCallee = "*(const _Float16 *)";
        mlir::Value aDCast =
            rewriter.create<emitc::CastOp>(loc, f16PtrType, al).getResult();
        mlir::Value aD =
            rewriter
                .create<emitc::CallOpaqueOp>(loc,
                                             mlir::TypeRange{f16ScalarType},
                                             f16ReadCallee,
                                             mlir::ValueRange{aDCast})
                .getResult(0);
        // The single activation scaled-sum s_y = *(const _Float16 *)&al.s (at
        // +2), folded into the LANE-WISE MIN term sumf += s_y * m_x. This is the
        // Family-B distinction from q4_0: the MIN correction term.
        step("act_sum_scalar");
        mlir::Value aSFull = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, al, sizeLit(activationSumOffset));
        mlir::Value aSCast =
            rewriter.create<emitc::CastOp>(loc, f16PtrType, aSFull).getResult();
        mlir::Value aS =
            rewriter
                .create<emitc::CallOpaqueOp>(loc,
                                             mlir::TypeRange{f16ScalarType},
                                             f16ReadCallee,
                                             mlir::ValueRange{aSCast})
                .getResult(0);

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
          step(vfwmulCallee);
          mlir::Value dC =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfwmulCallee,
                                               mlir::ValueRange{bDStrip, aD,
                                                                vl8})
                  .getResult(0);
          step(vfcvtCallee);
          mlir::Value sumiF =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfcvtCallee,
                                               mlir::ValueRange{sumiStrip, vl8})
                  .getResult(0);
          mlir::Value curF =
              rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfStrip)
                  .getResult();
          step(vfmaccCallee);
          mlir::Value scaled =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfmaccCallee,
                                               mlir::ValueRange{curF, sumiF, dC,
                                                                vl8})
                  .getResult(0);
          // MIN term m_x*s_y, added LANE-WISE.
          step(vfwmulCallee);
          mlir::Value mC =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfwmulCallee,
                                               mlir::ValueRange{bMStrip, aS,
                                                                vl8})
                  .getResult(0);
          step(vfaddCallee);
          mlir::Value nextF =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                               vfaddCallee,
                                               mlir::ValueRange{scaled, mC, vl8})
                  .getResult(0);
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
        step(vseCallee);
        rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, vseCallee,
                                             mlir::ValueRange{dst, sumfVal,
                                                              vl8});
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
      step(u8LoadCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8mf2Type},
                                       u8LoadCallee,
                                       mlir::ValueRange{cast, vl8})
          .getResult(0);
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
      step(reinterpretCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8mf2Type},
                                       reinterpretCallee, mlir::ValueRange{u})
          .getResult(0);
    };
    auto decodeLo = [&](mlir::Value packed) -> mlir::Value {
      step(vandCallee);
      mlir::Value mask =
          rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x0F").getResult();
      mlir::Value lo =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8mf2Type},
                                           vandCallee,
                                           mlir::ValueRange{packed, mask, vl8})
              .getResult(0);
      return reinterpretToI8(lo);
    };
    auto decodeHi = [&](mlir::Value packed) -> mlir::Value {
      step(vsrlCallee);
      mlir::Value four =
          rewriter.create<emitc::LiteralOp>(loc, immI32Type, "0x04").getResult();
      mlir::Value hi =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8mf2Type},
                                           vsrlCallee,
                                           mlir::ValueRange{packed, four, vl8})
              .getResult(0);
      return reinterpretToI8(hi);
    };
    // A scalar i8 read of the repacked activation quant byte a_ptr[l].qs[k].
    llvm::StringRef i8ReadCallee = "*(const int8_t *)";
    auto i8Read = [&](mlir::Value ab, mlir::Value byteOff) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, activationPtrType, ab, byteOff);
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      step("act_quant_scalar");
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       i8ReadCallee, mlir::ValueRange{cast})
          .getResult(0);
    };
    // vwmacc_vx widening multiply-accumulate: acc += scalar * vec (i8->i16).
    std::string vwmaccCallee = ("__riscv_vwmacc_vx_i16" + l16).str();
    auto vwmacc = [&](mlir::Value acc, mlir::Value scalar,
                      mlir::Value vec) -> mlir::Value {
      step(vwmaccCallee);
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                       vwmaccCallee,
                                       mlir::ValueRange{acc, scalar, vec, vl8})
          .getResult(0);
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
            step(fmvCallee);
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f")
                    .getResult();
            sumf[c] =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                                 fmvCallee,
                                                 mlir::ValueRange{zero, vl8})
                    .getResult(0);
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
              step(mvCallee);
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                               mvCallee,
                                               mlir::ValueRange{zero, vl8})
                  .getResult(0);
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
              step(vwaddCallee);
              sumi32[c] =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{i32m2Type},
                                                   vwaddCallee,
                                                   mlir::ValueRange{lo, hi, vl8})
                      .getResult(0);
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
              step(f16LoadCallee);
              return rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f16m1Type},
                                               f16LoadCallee,
                                               mlir::ValueRange{dCast, vl8})
                  .getResult(0);
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
              step("act_scale_scalar");
              mlir::Value aDOff = rewriter.create<emitc::AddOp>(
                  loc, activationPtrType, al, sizeLit(c * 2));
              mlir::Value aDCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, aDOff)
                      .getResult();
              mlir::Value aD =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{f16ScalarType}, f16ReadCallee,
                          mlir::ValueRange{aDCast})
                      .getResult(0);
              // Scale term (d_x*d_y_c)*sumi_c.
              step(vfwmulCallee);
              mlir::Value dC =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{f32m2Type},
                                                   vfwmulCallee,
                                                   mlir::ValueRange{bD, aD, vl8})
                      .getResult(0);
              step(vfcvtCallee);
              mlir::Value sumiF =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{f32m2Type},
                                                   vfcvtCallee,
                                                   mlir::ValueRange{sumi32[c],
                                                                    vl8})
                      .getResult(0);
              mlir::Value curF =
                  rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumfVar[c])
                      .getResult();
              step(vfmaccCallee);
              mlir::Value scaled =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{f32m2Type}, vfmaccCallee,
                          mlir::ValueRange{curF, sumiF, dC, vl8})
                      .getResult(0);
              // s_y_c = *(const _Float16 *)&al.s[c]  (al + 8 + c*2).
              step("act_sum_scalar");
              mlir::Value aSOff = rewriter.create<emitc::AddOp>(
                  loc, activationPtrType, al,
                  sizeLit(activationSumOffset + c * 2));
              mlir::Value aSCast =
                  rewriter.create<emitc::CastOp>(loc, f16PtrType, aSOff)
                      .getResult();
              mlir::Value aS =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{f16ScalarType}, f16ReadCallee,
                          mlir::ValueRange{aSCast})
                      .getResult(0);
              // MIN term m_x*s_y_c, added LANE-WISE.
              step(vfwmulCallee);
              mlir::Value mC =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{f32m2Type},
                                                   vfwmulCallee,
                                                   mlir::ValueRange{bM, aS, vl8})
                      .getResult(0);
              step(vfaddCallee);
              mlir::Value nextF =
                  rewriter
                      .create<emitc::CallOpaqueOp>(loc,
                                                   mlir::TypeRange{f32m2Type},
                                                   vfaddCallee,
                                                   mlir::ValueRange{scaled, mC,
                                                                    vl8})
                      .getResult(0);
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
            step(vseCallee);
            rewriter.create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{}, vseCallee,
                mlir::ValueRange{dst, sumfVal, vl8});
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

mlir::LogicalResult VariantToEmitCFunc::emitQ4_1Q8_1BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
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

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-half-block anchor). q4_1's nibble
    // half-block matches q4_0's, so the anchor set is q4_0's {mf4, m1}. It is the
    // *how* (vector grouping / strip width), never the *what*: the dot product is
    // byte-exact either way. The chosen i8 source LMUL drives the widened i16
    // product LMUL and the load/vsetvl spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (mf4->mf2, m1->m2).
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    // The UNSIGNED weight lane the unsigned-nibble decode runs on (q4_1-only;
    // the activations stay i8).
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t weightMinOffset = blockDot.getWeightMinByteOffset();
    int64_t activationSumOffset = blockDot.getActivationSumByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*20;
    // const uint8_t *yb = vy + (ib+blockOffset)*36.
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

    // A scalar fp16->fp32 read at a per-block byte offset (the ONE sanctioned
    // opaque piece, a typed emitc.call_opaque node). q4_1 reads FOUR per block:
    // d_x at weight+0, m_x at weight+`weight_min_byte_offset`; d_y at
    // activation+0, s_y at activation+`activation_sum_byte_offset`.
    auto fp16ReadAt = [&](mlir::Value blockBase, mlir::Type ptrType,
                          int64_t byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, ptrType, blockBase,
                                             sizeLit(byteOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{addr})
          .getResult(0);
    };

    // The UNSIGNED-nibble asymmetric i4xi8 decode/product + reduce for ONE strip,
    // seeded with the carried sumi, returning the next scalar sumi. The weight
    // chunk is loaded as UNSIGNED (vle8_v_u8<L>) so the vand/vsrl logical decode
    // is correct; the activations stay i8. vl is the active strip VL; chunkOffset
    // is the within-half-block byte offset (the strip induction var for the
    // robust loop, or a 0 literal for the elided core).
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
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
      // The weight load is UNSIGNED (vle8_v_u8<L>); the q8 activations are i8.
      std::string weightLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      std::string actLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadU8 = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, weightLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                         weightLoadCallee,
                                         mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      auto loadI8 = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, actLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         actLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      mlir::Value w =
          loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
      mlir::Value y0 =
          loadI8(chunkPtr(yb, activationPtrType, i8PtrType, quantOffset));
      mlir::Value y1 = loadI8(
          chunkPtr(yb, activationPtrType, i8PtrType, quantOffset + highOffset));

      mlir::FailureOr<mlir::Value> product =
          emitUnsignedNibbleDecodeProductValue(rewriter, loc, w, y0, y1, vl,
                                               i8CoreType, u8CoreType,
                                               i16WideType, coreLmul, 16,
                                               wideLmul, "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce, returns the sumi lvalue. When forceRobust is set the inner strip
    // loop is always kept; otherwise stripElided selects the single-vsetvl(16)
    // elided core (m1-only, VLEN >= 128).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) + ONE strip
        // reduce. NO inner strip loop, NO sumi carry.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, sizeLit(0), vl, sumiVar, /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 16 weight bytes; the loop STEP
      // is the loop-invariant VLMAX, the per-chunk active vl is vsetvl(16 - c).
      // Stays VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, c, vl, sumiVar, /*carrySumi=*/true);
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
    };

    // The fp32 accumulate sumf = sumf + (d_x*d_y)*sumi + m_x*s_y (ggml q4_1 EXACT
    // statement, quants.c:319), grouped into ONE emitc.expression so
    // mlir-translate renders it as a SINGLE C statement matching ggml's fp
    // associativity / contraction byte-for-byte. The two summed products
    // (the scale term and the MIN term) are the Family-B distinction from q4_0's
    // single block term. The emitc.load temps stay OUTSIDE the expression (load
    // lacks the CExpression trait). For multi-block unrolls the caller invokes
    // this in STRICT ascending block order, preserving fp non-associativity.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY,
                        mlir::Value mX, mlir::Value sY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // (d_x * d_y) FIRST, then * sumi -- ggml q4_1's scale term order.
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value scaleTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scaleProduct,
                                          sumiFloat);
        // m_x * s_y -- the per-block MIN/SUM correction term.
        mlir::Value minTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, mX, sY);
        // ggml's EXACT tree (quants.c:319): `sumf += (d_x*d_y)*sumi + m_x*s_y;`.
        // The `+` between the two products binds BEFORE the `+=`, so the two
        // products are summed FIRST, then added to sumf: sumf + (scaleTerm +
        // minTerm). This grouping is fp-significant (`(sumf+A)+B != sumf+(A+B)`)
        // -- byte-exactness requires this exact tree.
        mlir::Value blockTerm =
            rewriter.create<emitc::AddOp>(loc, floatType, scaleTerm, minTerm);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block's integer core (addresses + four scales + sumi), WITHOUT
    // the fold; the caller folds in strict ascending block order.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
      mlir::Value mX;
      mlir::Value sY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      // ggml reads d_x then d_y for the scale product, then m_x and s_y for the
      // MIN term; read them in that order for a stable node sequence.
      mlir::Value dX = fp16ReadAt(xb, weightPtrType, 0);
      mlir::Value dY = fp16ReadAt(yb, activationPtrType, 0);
      mlir::Value mX = fp16ReadAt(xb, weightPtrType, weightMinOffset);
      mlir::Value sY = fp16ReadAt(yb, activationPtrType, activationSumOffset);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY, mX, sY};
    };

    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
    } else {
      // Multi-block unroll: emit ALL factor independent integer cores FIRST (so
      // they overlap), THEN the factor folds in strict ascending block order,
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
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY, core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
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

mlir::LogicalResult VariantToEmitCFunc::emitQ8_0Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
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

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (m2 default
    // to match ggml's one-vwredsum-per-block anchor, or m1/mf4 to re-strip the
    // 32-element block). It is the *how* (vector grouping / strip width), never
    // the *what*: the dot product is byte-exact for any anchor (vwredsum sums the
    // same integer set; integer add is order-independent). The chosen i8 source
    // LMUL drives the widened i16 product LMUL and the i8 load/vsetvl spelling.
    llvm::StringRef coreLmul = "m2";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (m2->m4, m1->m2,
    // mf4->mf2). The verifier bounds the source to mf4|m1|m2.
    llvm::StringRef wideLmul =
        (coreLmul == "m2") ? "m4" : (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();
    // q8_0 has NO high-half split: the integer core spans the whole QK=32-element
    // block, not the Q4_0 sibling's 16-lane half-block.
    int64_t blockLen = qk;

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*34;
    // const uint8_t *yb = vy + (ib+blockOffset)*34 (both operands block_q8_0).
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

    // The two scalar fp16->fp32 reads (the ONE sanctioned opaque piece).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // The plain i8 x i8 widening product + reduce for ONE strip, seeded with the
    // carried sumi, returning the next scalar sumi. vl is the active strip VL;
    // chunkOffset is the within-block byte offset (the strip induction var for
    // the robust loop, or a 0 literal for the elided core). When carrySumi is set
    // the seed lane0 is loaded from sumiVar; otherwise it is a "0" literal.
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      };
      std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadChunk = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         loadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      mlir::Value vx0 = loadChunk(chunkPtr(xb, weightPtrType, quantOffset));
      mlir::Value vy0 = loadChunk(chunkPtr(yb, activationPtrType, quantOffset));

      // Plain signed widening product: i8 x i8 -> i16 (NO nibble decode).
      std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value product =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16WideType},
                                           mulCallee,
                                           mlir::ValueRange{vx0, vy0, vl})
              .getResult(0);

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce over the whole 32-element block, returns the sumi lvalue. When
    // forceRobust is set the inner strip loop is always kept (the tail / robust
    // shapes); otherwise stripElided selects the single-vsetvl(32) elided core.
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The vsetvl SEW/LMUL spelling of the anchor: m2/m1 use vsetvl_e8<lmul>
      // (byte strips), mf4 uses vsetvl_e32m1 (4-element strips at VLEN=128).
      unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
      llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");

      if (!forceRobust && stripElided) {
        // Elided core (m2-only, VLEN >= 128): ONE vsetvl_e8m2(32) (caps the
        // active vl at 32 when VLMAX >= 32, covering the whole block) + ONE
        // strip reduce. NO inner strip loop, NO sumi carry (seed lane0 = 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(blockLen)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, sizeLit(0), vl, sumiVar, /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 32 block bytes; the loop STEP
      // is the loop-invariant VLMAX, the per-chunk active vl is vsetvl(32 - c).
      // Stays VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(blockLen)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(blockLen), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, c, vl, sumiVar, /*carrySumi=*/true);
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
    };

    // The fp32 accumulate sumf = sumf + (float)sumi * (d_x * d_y) (ggml's EXACT
    // q8_0 order: the two scales are multiplied FIRST, then by the integer sum;
    // distinct from the Q4_0 sibling's left-assoc ((sumi*d_x)*d_y)). Grouped into
    // ONE emitc.expression so mlir-translate renders it as a SINGLE C statement
    // and the compiler fuses the SAME FMA ggml does under -ffp-contract=on/
    // default. For multi-block unrolls the caller invokes this in STRICT
    // ascending block order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // d_x * d_y FIRST (ggml q8_0 order).
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat,
                                          scaleProduct);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block's integer core (addresses + scales + sumi), WITHOUT the
    // fold; the caller folds in strict ascending block order.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      mlir::Value dX = fp16Read(xb);
      mlir::Value dY = fp16Read(yb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY};
    };

    if (multiBlockFactor == 1) {
      // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the no-unroll form.
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY);
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
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
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

    // The binary sign decode anchors at the m1 8-lane sign-plane groups (the
    // verifier fixes integer_core_lmul to m1). i8m1 source -> the i16m2 widened
    // product the vwredsum reduces into i32m1.
    llvm::StringRef coreLmul = "m1";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type u8CoreType =
        emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");

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
    int64_t groupLanes = 8;                              // bits per byte / sign group

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The kmask bit-selector {1<<m} is an inline const (a trivial bit-position
    // vector), emitted as a decl + broadcast load ONCE above the block loop.
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_q1_0_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

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

    // vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_q1_0_kmask, 8);  (ONCE)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "kmask_table_load"));
    mlir::Value kmaskName =
        rewriter.create<emitc::LiteralOp>(loc, u8PtrType, "tcrv_q1_0_kmask");
    mlir::Value kmaskVec =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType},
                riscvIntrinsicName("vle", 8, coreLmul, "u8"),
                mlir::ValueRange{kmaskName, sizeLit(groupLanes)})
            .getResult(0);

    // The ONE sanctioned opaque scalar fp16->fp32 read (a typed emitc.call_opaque
    // node, exactly how the q4_0 sibling emits its fp16 scale reads).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
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

        // int32_t sumi_block = 0;  (RESET each sub-block; the integer dot)
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi_block", opName, role));
        auto sumiBlockVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiBlockVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // size_t vl = __riscv_vsetvl_e8m1(8);  (the 8-lane sign-plane group)
        std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, coreLmul, "");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, setvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             setvlCallee,
                                             mlir::ValueRange{sizeLit(groupLanes)})
                .getResult(0);

        // The FOUR 8-lane sign-plane groups of this q8 sub-block (32 lanes),
        // carrying sumi_block across groups.
        for (int64_t b = 0; b < bytesPerSubBlock; ++b) {
          // uint8_t bits = *(const uint8_t *)(xb + 2 + k*4 + b);  (scalar read,
          // the structured cast + subscript[0] + load idiom, exactly how mxfp4
          // reads its E8M0 exponent byte).
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "bits_byte_read"));
          int64_t bitByteOffset = weightQuantOffset + k * bytesPerSubBlock + b;
          mlir::Value bitsPtr = rewriter
                                    .create<emitc::AddOp>(loc, weightPtrType, xb,
                                                          sizeLit(bitByteOffset))
                                    .getResult();
          mlir::Value bitsPtrU8 =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, bitsPtr).getResult();
          mlir::Value bitsIdx0 =
              rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(),
                                                "0");
          mlir::Value bitsElem =
              rewriter
                  .create<emitc::SubscriptOp>(
                      loc,
                      mlir::cast<mlir::TypedValue<emitc::PointerType>>(bitsPtrU8),
                      bitsIdx0)
                  .getResult();
          mlir::Value bits =
              rewriter.create<emitc::LoadOp>(loc, constU8Type, bitsElem)
                  .getResult();

          // m_bits = vand(vmv_v_x(bits), kmask); m = vmsne(m_bits, 0).
          std::string bcastCallee = riscvIntrinsicName("vmv_v_x", 8, coreLmul, "u8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, bcastCallee));
          mlir::Value bitsBcast =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                               bcastCallee,
                                               mlir::ValueRange{bits, vl})
                  .getResult(0);
          std::string andCallee = riscvIntrinsicName("vand", 8, coreLmul, "u8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, andCallee));
          mlir::Value bitSel =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{u8CoreType}, andCallee,
                      mlir::ValueRange{bitsBcast, kmaskVec, vl})
                  .getResult(0);
          std::string msneCallee =
              riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, msneCallee));
          mlir::Value zeroLit = rewriter.create<emitc::LiteralOp>(
              loc, emitc::OpaqueType::get(ctx, "int"), "0");
          mlir::Value signMask =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{maskType}, msneCallee,
                      mlir::ValueRange{bitSel, zeroLit, vl})
                  .getResult(0);

          // vint8m1_t q8 = vle8(yb + 2 + b*8);  (the 8 q8 lanes of this group)
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "q8_group_addr"));
          int64_t q8GroupOffset = activationQuantOffset + b * groupLanes;
          mlir::Value q8Ptr = rewriter
                                  .create<emitc::AddOp>(loc, activationPtrType, yb,
                                                        sizeLit(q8GroupOffset))
                                  .getResult();
          mlir::Value q8PtrI8 =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Ptr).getResult();
          std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, loadCallee));
          mlir::Value q8 =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                               loadCallee,
                                               mlir::ValueRange{q8PtrI8, vl})
                  .getResult(0);

          // w0 = vwcvt_x_x_v_i16m2(q8);  WIDEN to i16 FIRST, then negate/merge in
          // the i16 domain. ggml's _generic negates in the INT domain, so -q8 of
          // the int8 boundary value -128 is +128 -- which does NOT fit i8 (vneg on
          // i8 -128 overflows back to -128). Performing the vneg/vmerge AFTER the
          // i8->i16 widening keeps +128 representable, so the decode is byte-exact
          // over the FULL int8 range (not just ggml's [-127,127] q8 domain).
          std::string wcvtCallee =
              riscvIntrinsicName("vwcvt_x_x_v", 16, wideLmul, "i16");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, wcvtCallee));
          mlir::Value q8Wide =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16WideType},
                                               wcvtCallee,
                                               mlir::ValueRange{q8, vl})
                  .getResult(0);

          // sgn = vmerge(vneg_i16(q8w), q8w, m);  (+q8 where bit set, -q8 where
          // clear). The vbool8_t sign mask (from the u8m1 8-lane vmsne) is the
          // SAME predicate width the i16m2 vmerge expects (SEW/LMUL = 16/2 = 8).
          std::string negCallee = "__riscv_vneg_v_i16" + wideLmul.str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, negCallee));
          mlir::Value q8WideNeg =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16WideType},
                                               negCallee,
                                               mlir::ValueRange{q8Wide, vl})
                  .getResult(0);
          std::string mergeCallee =
              riscvIntrinsicName("vmerge", 16, wideLmul, "i16");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, mergeCallee));
          mlir::Value widened =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i16WideType}, mergeCallee,
                      mlir::ValueRange{q8WideNeg, q8Wide, signMask, vl})
                  .getResult(0);

          // seed lane0 = sumi_block, vwredsum, extract back into sumi_block.
          std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seedCallee));
          mlir::Value sumiSeed =
              rewriter.create<emitc::LoadOp>(loc, i32Type, sumiBlockVar)
                  .getResult();
          mlir::Value seed =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                               seedCallee,
                                               mlir::ValueRange{sumiSeed,
                                                                sizeLit(1)})
                  .getResult(0);
          std::string reduceCallee =
              ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reduceCallee));
          mlir::Value red =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                               reduceCallee,
                                               mlir::ValueRange{widened, seed,
                                                                vl})
                  .getResult(0);
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extractCallee));
          mlir::Value next =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                               extractCallee,
                                               mlir::ValueRange{red})
                  .getResult(0);
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi_block", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiBlockVar, next);
        }

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

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
