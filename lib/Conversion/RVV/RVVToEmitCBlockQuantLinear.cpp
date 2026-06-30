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

    // The integer core's LMUL is a bounded resource/scheduling fact. The DEFAULT
    // is "m1" -- the WIDE-LMUL whole-half-block anchor that matches ggml's
    // one-vwredsum-per-block reduction (16-lane half-block in ONE strip at
    // VLEN>=128). m1 is the VLEN-UNIVERSAL FLOOR: the m1 robust strip loop is
    // byte-correct at ANY VLEN (a narrow VLMAX just re-strips with the sumi-carry
    // seed), so an attr-less op lowers to the wide form on every target, exactly
    // like q8_0's "m2" floor and the codebook's "m1" floor. The gearbox REFINES
    // it -- it stamps integer_core_lmul (and the elision/factor knobs) from
    // getRVVStripVLMAXElements (the single truth source), e.g. (m1, factor=4,
    // elided) at a guaranteed Zvl128b target; the legacy "mf4" narrow anchor is
    // still a stampable candidate (the now-non-default 4x8-lane sub-VLMAX form),
    // never the default. LMUL is the *how* (vector grouping / strip width), never
    // the *what*: the dot product is byte-exact across anchors (vwredsum sums the
    // same integer set; integer add is order-independent). The chosen i8 source
    // LMUL ("m1"/"mf4") drives the widened i16 product LMUL ("m2"/"mf2") and the
    // i8 load/vsetvl spelling. The two outer-loop SHAPE knobs (also bounded
    // resource/scheduling facts, byte-exact -- the fp32 folds stay in strict
    // ascending block order, only the independent integer cores overlap). The
    // verifier bounds the factor to 1|2|4 and the elision to robust|elided, and
    // forbids "elided" unless the integer core anchors at m1.
    BlockDotFacts blockDotFacts = deriveBlockDotFacts(blockDot, "m1");
    llvm::StringRef coreLmul = blockDotFacts.coreLmul;
    int64_t multiBlockFactor = blockDotFacts.multiBlockFactor;
    bool stripElided = blockDotFacts.stripElided;
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
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
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
        return emitOpaqueCall(rewriter, loc, i8CoreType, loadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
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
      // vwredsum is LMUL-independent in result: the i16<wide> source reduces
      // into i32m1 lane 0 (the same integer set; the callee tracks the source
      // LMUL).
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red =
          emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                         mlir::ValueRange{*product, seed, vl}, opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
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
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(halfBlock)};
            });
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
      mlir::Value innerVlmax = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(halfBlock)};
          });
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        // size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining = rewriter.create<emitc::SubOp>(
                  loc, sizeType, sizeLit(halfBlock), c);
              return {remaining};
            });

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
    BlockDotFacts blockDotFacts = deriveBlockDotFacts(blockDot, "mf4");
    llvm::StringRef coreLmul = blockDotFacts.coreLmul;
    int64_t multiBlockFactor = blockDotFacts.multiBlockFactor;
    bool stripElided = blockDotFacts.stripElided;
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
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
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
        return emitOpaqueCall(rewriter, loc, u8CoreType, wLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, yLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
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
                         mlir::ValueRange{*product, seed, vl}, opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
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
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(halfBlock)};
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

      // Robust core: the inner strip loop over the 16 weight bytes. The loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). VLEN-robust via the sumi-carrying seed.
      mlir::Value innerVlmax = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(halfBlock)};
          });
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
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
                  loc, sizeType, sizeLit(halfBlock), c);
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
    BlockDotFacts blockDotFacts = deriveBlockDotFacts(blockDot, "mf4");
    llvm::StringRef coreLmul = blockDotFacts.coreLmul;
    int64_t multiBlockFactor = blockDotFacts.multiBlockFactor;
    bool stripElided = blockDotFacts.stripElided;
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
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{addr}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
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
        return emitOpaqueCall(rewriter, loc, u8CoreType, wLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, yLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
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
                         mlir::ValueRange{*product, seed, vl}, opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
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
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(halfBlock)};
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

      // Robust core: the inner strip loop over the 16 weight bytes. The loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). VLEN-robust via the sumi-carrying seed.
      mlir::Value innerVlmax = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(halfBlock)};
          });
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
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
                  loc, sizeType, sizeLit(halfBlock), c);
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

    // The integer core's LMUL is a bounded resource/scheduling fact. The DEFAULT
    // is "m1" -- the WIDE-LMUL whole-half-block anchor matching ggml's
    // one-vwredsum-per-half-block reduction (16-lane half-block, ONE strip at
    // VLEN>=128). q4_1's nibble half-block matches q4_0's, so the anchor set is
    // q4_0's {mf4, m1} and the SAME wide-universal-floor reasoning applies: the m1
    // robust strip is byte-correct at any VLEN, so an attr-less op lowers wide;
    // the gearbox refines the elision/factor knobs and the legacy "mf4" narrow
    // anchor stays a stampable (now non-default) candidate. LMUL is the *how*
    // (vector grouping / strip width), never the *what*: the dot product is
    // byte-exact either way. The chosen i8 source LMUL drives the widened i16
    // product LMUL and the load/vsetvl spelling.
    BlockDotFacts blockDotFacts = deriveBlockDotFacts(blockDot, "m1");
    llvm::StringRef coreLmul = blockDotFacts.coreLmul;
    int64_t multiBlockFactor = blockDotFacts.multiBlockFactor;
    bool stripElided = blockDotFacts.stripElided;
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
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{addr}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
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
        return emitOpaqueCall(rewriter, loc, u8CoreType, weightLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      auto loadI8 = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, actLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
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
                         mlir::ValueRange{*product, seed, vl}, opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
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
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(halfBlock)};
            });
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
      mlir::Value innerVlmax = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(halfBlock)};
          });
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
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
                  loc, sizeType, sizeLit(halfBlock), c);
              return {remaining};
            });

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
    BlockDotFacts blockDotFacts = deriveBlockDotFacts(blockDot, "m2");
    llvm::StringRef coreLmul = blockDotFacts.coreLmul;
    int64_t multiBlockFactor = blockDotFacts.multiBlockFactor;
    bool stripElided = blockDotFacts.stripElided;
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
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
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
        return emitOpaqueCall(rewriter, loc, i8CoreType, loadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      mlir::Value vx0 = loadChunk(chunkPtr(xb, weightPtrType, quantOffset));
      mlir::Value vy0 = loadChunk(chunkPtr(yb, activationPtrType, quantOffset));

      // Plain signed widening product: i8 x i8 -> i16 (NO nibble decode).
      std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
      mlir::Value product =
          emitOpaqueCall(rewriter, loc, i16WideType, mulCallee,
                         mlir::ValueRange{vx0, vy0, vl}, opName, role);

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
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
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

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
