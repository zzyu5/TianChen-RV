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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace detail {

// VariantToEmitCFunc forward-pass elementwise emit methods: ggml vec_scale /
// rmsnorm / silu (+ vexpf) / softmax / quantize_row_q8_0 / rope_norm. Split out
// of RVVToEmitC.cpp as a pure code move; the emitted C is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitTypedElementwiseLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The M-FLAT forward-elementwise scaffold's typed strip-loop lowering, the
    // constructed sibling of emitTypedFlatBlockDotLoopBody. The outer strip loop
    // is owned by the loop op; the per-strip map is re-emitted from the region's
    // core brick (anti-bypass: the brick's strip_index MUST be the loop induction
    // variable). BYTE-EXACT to the retired monolith tcrv_rvv.ggml_vec_scale_f32
    // emit modulo ONLY the source-op provenance token.
    tcrvrvv::TypedElementwiseLoopBodyOp loopBody;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto lb = llvm::dyn_cast<tcrvrvv::TypedElementwiseLoopBodyOp>(op))
        loopBody = lb;
    }
    if (!loopBody)
      return rewriter.notifyMatchFailure(
          scope, "typed elementwise loop body missing the op");

    // The "reduce" model carries a loop-carried accumulator: the FIRST forward
    // REDUCE operator constructed through the scaffold is rms_norm (the Σx²
    // scalar-double fold + the scalar rsqrt + the vectorized normalize strip),
    // whose per-element fold rides the tcrv_rvv.elementwise_rms_norm_reduce_core
    // reduce-core brick. Dispatch to its dedicated re-emit before the map path.
    if (loopBody.getReduceMapModel() == "reduce") {
      tcrvrvv::ElementwiseRmsNormReduceCoreOp rmsCore;
      loopBody.getBody().walk([&](tcrvrvv::ElementwiseRmsNormReduceCoreOp o) {
        rmsCore = o;
      });
      if (!rmsCore)
        return rewriter.notifyMatchFailure(
            loopBody, "reduce-model elementwise loop body requires a recognized "
                      "reduce core brick (elementwise_rms_norm_reduce_core)");
      return emitElementwiseRmsNormReduceStrip(rewriter, loc, loopBody, rmsCore,
                                               avlArg, sizeType, valueMap);
    }

    // The "rotate" model is the per-PAIR scalar recurrence shape: the FIRST (and
    // only) forward ROTATE operator constructed through the scaffold is rope
    // (the position-dependent 2x2 rotation on consecutive pairs + the scalar-libm
    // cos/sin angle seam + the f32 theta recurrence), whose per-pair work rides
    // the tcrv_rvv.elementwise_rope_rotate_core rotate-core brick. Dispatch to its
    // dedicated re-emit before the map path.
    if (loopBody.getReduceMapModel() == "rotate") {
      tcrvrvv::ElementwiseRopeRotateCoreOp ropeCore;
      loopBody.getBody().walk([&](tcrvrvv::ElementwiseRopeRotateCoreOp o) {
        ropeCore = o;
      });
      if (!ropeCore)
        return rewriter.notifyMatchFailure(
            loopBody, "rotate-model elementwise loop body requires a recognized "
                      "rotate core brick (elementwise_rope_rotate_core)");
      return emitElementwiseRopeRotateStrip(rewriter, loc, loopBody, ropeCore,
                                            avlArg, sizeType, valueMap);
    }

    // The "map" model's core brick is a per-strip elementwise map: the scale map
    // (elementwise_scale_map, y[i] *= v, in-place single buffer) or the silu map
    // (elementwise_silu_map, y[i] = x[i]*sigmoid(x[i]), a two-buffer x->y map).
    // Find the brick + yield.
    tcrvrvv::ElementwiseScaleMapOp mapOp;
    tcrvrvv::ElementwiseSiluMapOp siluOp;
    tcrvrvv::TypedElementwiseLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<tcrvrvv::ElementwiseScaleMapOp>(bodyOp))
        mapOp = o;
      else if (auto o = llvm::dyn_cast<tcrvrvv::ElementwiseSiluMapOp>(bodyOp))
        siluOp = o;
      else if (auto o =
                   llvm::dyn_cast<tcrvrvv::TypedElementwiseLoopYieldOp>(bodyOp))
        yieldOp = o;
    });
    if (!yieldOp || (!mapOp && !siluOp))
      return rewriter.notifyMatchFailure(
          loopBody, "map-model elementwise loop body requires a recognized map "
                    "core brick (elementwise_scale_map | elementwise_silu_map) + "
                    "the loop yield");

    // The SILU map reuses the SAME outer strip-loop op + map model, but its
    // per-strip decode is the m2 exp polynomial over two buffers (x->y), so it
    // owns a dedicated core-brick emit. Dispatch to it before the scale path.
    if (siluOp)
      return emitElementwiseSiluMapStrip(rewriter, loc, loopBody, siluOp, avlArg,
                                         sizeType, valueMap);

    // Anti-bypass (I7): the brick's strip_index MUST be the loop induction
    // variable (region arg 0), so the emit addresses buffer + i, not strip 0.
    mlir::Value stripIndex = loopBody.getBody().front().getArgument(0);
    if (mapOp.getStripIndex() != stripIndex)
      return rewriter.notifyMatchFailure(
          mapOp, "the elementwise_scale_map brick's strip_index must be the "
                 "loop induction variable (region arg 0)");

    // The ABI bases are sourced from the BRICK's operands (not the loop op), the
    // same anti-bypass convention the flat block-dot bricks use.
    mlir::Value buffer = valueMap.lookup(mapOp.getBuffer());
    mlir::Value scalar = valueMap.lookup(mapOp.getScalar());
    if (!buffer || !scalar)
      return rewriter.notifyMatchFailure(mapOp, "scale ABI operand unmapped");

    llvm::StringRef opName = mapOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = mapOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type bufferPtrType = buffer.getType();

    // The f32 strip-loop LMUL is a bounded resource/scheduling fact (default m8,
    // matching ggml's hand-written path). It is the *how* (vector grouping /
    // strip width), never the *what*: the result is byte-exact at any anchor
    // (every lane is multiplied by the same scalar v). The verifier bounds it to
    // m1|m2|m4|m8 (carried as "strip_lmul" so the op holds no forbidden
    // dataflow-parameter "lmul" at the I5 boundary).
    llvm::StringRef lmul = mapOp.getStripLmul().value_or("m8");
    std::string f32VecTypeName = ("vfloat32" + lmul + "_t").str();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, f32VecTypeName);
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m<L>(n).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    mlir::Value vlmax = emitOpaqueCall(rewriter, loc, sizeType, setvlCallee,
                                       mlir::ValueRange{avlArg}, opName, role);

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // Remaining-AVL setvl: size_t vl = __riscv_vsetvl_e32m<L>(n - i).
      mlir::Value bodyVL = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, setvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value remaining =
                b.create<emitc::SubOp>(l, sizeType, avlArg, inductionVar);
            return {remaining};
          });

      // In-place element pointer: float *p = y + i.
      mlir::Value elemPtr = rewriter.create<emitc::AddOp>(
          loc, bufferPtrType, buffer, inductionVar);
      mlir::Value loadPtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, elemPtr).getResult();

      // vfloat32m<L>_t ay = __riscv_vle32_v_f32m<L>(p, vl);
      std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
      mlir::Value ay = emitOpaqueCall(rewriter, loc, f32VecType, loadCallee,
                                      mlir::ValueRange{loadPtr, bodyVL}, opName,
                                      role);

      // vfloat32m<L>_t ny = __riscv_vfmul_vf_f32m<L>(ay, v, vl);  scalar bcast.
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      mlir::Value ny = emitOpaqueCall(rewriter, loc, f32VecType, mulCallee,
                                      mlir::ValueRange{ay, scalar, bodyVL},
                                      opName, role);

      // __riscv_vse32_v_f32m<L>(p, ny, vl);  store back in place.
      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");
      emitOpaqueCallVoid(rewriter, loc, storeCallee,
                         mlir::ValueRange{loadPtr, ny, bodyVL}, opName, role);
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseRmsNormReduceStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::TypedElementwiseLoopBodyOp loopBody,
    tcrvrvv::ElementwiseRmsNormReduceCoreOp rmsCore, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The CONSTRUCTED rms_norm reduce-model body, the reduce sibling of the map
    // paths (scale/silu) and of the block-dot loop scaffold. The outer loop op
    // owns the reduce shape (reduce_map_model "reduce": a loop-carried f64
    // accumulator region arg + the yield that carries it back); this re-emit
    // sources the WHOLE rms_norm ABI + the byte-exact scalar-double Σx² fold /
    // scalar rsqrt / vectorized normalize strip from the region's reduce core
    // brick (anti-bypass). BYTE-EXACT to the retired monolith
    // tcrv_rvv.ggml_rms_norm_f32 emit modulo ONLY the source-op provenance token.

    // Anti-bypass (I7): the brick's strip_index MUST be the loop induction
    // variable (region arg 0) and its acc MUST be the loop-carried accumulator
    // (region arg 1); the verifier pins both, checked here fail-closed too.
    mlir::Block &block = loopBody.getBody().front();
    if (block.getNumArguments() < 2 ||
        rmsCore.getStripIndex() != block.getArgument(0) ||
        rmsCore.getAcc() != block.getArgument(1))
      return rewriter.notifyMatchFailure(
          rmsCore, "the rms_norm reduce core brick's strip_index / acc must be "
                   "the loop induction variable / loop-carried accumulator "
                   "(region args 0 / 1)");

    mlir::Value input = valueMap.lookup(rmsCore.getInput());
    mlir::Value outputBuf = valueMap.lookup(rmsCore.getOutput());
    mlir::Value eps = valueMap.lookup(rmsCore.getEps());
    if (!input || !outputBuf || !eps)
      return rewriter.notifyMatchFailure(rmsCore,
                                         "rms_norm ABI operand unmapped");

    llvm::StringRef opName = rmsCore.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = rmsCore.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = outputBuf.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type doubleType = emitc::OpaqueType::get(ctx, "double");
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // ggml_float sum = 0.0;  (the SCALAR double accumulator -- ggml_float is
    // double; ops.cpp:3791). emitc.for has no iter_args, so the loop-carried
    // accumulator is an emitc.variable lvalue + emitc.assign, exactly as the
    // block-dot kernels carry the fp32 *s accumulator.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sum", opName, role));
    auto sumVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(doubleType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumVar, rewriter.create<emitc::LiteralOp>(loc, doubleType, "0.0"));

    // for (size_t i = 0; i < ne00; ++i) { ... }  -- the SCALAR ascending fold
    // (step 1). This loop is NOT vectorized: a vectorized vfredusum would fold in
    // f32 with a tree order and break byte-exactness vs ggml.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "scalar_double_reduce"));
    mlir::Value zeroIdx =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value oneStep =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    auto reduceFor = rewriter.create<emitc::ForOp>(loc, zeroIdx, avlArg, oneStep,
                                                   /*bodyBuilder=*/nullptr);
    mlir::Value redIdx = reduceFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(reduceFor.getBody());

      // const float *xp = (const float *)(x + i);  float xi = xp[0];
      mlir::Value xElemPtr = rewriter.create<emitc::AddOp>(
          loc, inputPtrType, input, redIdx);
      auto xElemPtrCast =
          llvm::cast<mlir::TypedValue<emitc::PointerType>>(
              rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xElemPtr)
                  .getResult());
      mlir::Value xElemIndex =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      emitc::SubscriptOp xSubscript =
          rewriter.create<emitc::SubscriptOp>(loc, xElemPtrCast, xElemIndex);
      auto xLValueType =
          llvm::cast<emitc::LValueType>(xSubscript.getResult().getType());
      mlir::Value xiLoaded =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     xSubscript.getResult())
              .getResult();

      // The load-bearing cast chain, grouped into ONE emitc.expression so
      // mlir-translate renders ONE C statement: sum = sum + (double)(xi * xi).
      // The f32 product rounds FIRST, is WIDENED to double, then added in double.
      // The widen sits BETWEEN the f32 multiply and the double add -- an FMA
      // barrier (different types), so -ffp-contract cannot fuse the two. The
      // emitc.load temps stay OUTSIDE the expression (load lacks the CExpression
      // trait).
      mlir::Value sumCur =
          rewriter.create<emitc::LoadOp>(loc, doubleType, sumVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, doubleType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        // float p = xi * xi;  (f32 product -- one f32 rounding)
        mlir::Value prod =
            rewriter.create<emitc::MulOp>(loc, floatType, xiLoaded, xiLoaded);
        // (double)p  -- widen the f32 product to double (the FMA barrier).
        mlir::Value prodWide =
            rewriter.create<emitc::CastOp>(loc, doubleType, prod).getResult();
        // sum + (double)p  -- accumulate in double.
        mlir::Value sumNext =
            rewriter.create<emitc::AddOp>(loc, doubleType, sumCur, prodWide);
        rewriter.create<emitc::YieldOp>(loc, sumNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sum", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumVar, accumExpr.getResult());
    }

    // float mean = (float)(sum / (double)ne00);  -- divide in DOUBLE, cast to
    // f32 AFTER the division (ggml's `const float mean = sum/ne00;` with
    // ne00 promoted to double for the ggml_float divide; ops.cpp:3797).
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "mean"));
    mlir::Value sumFinal =
        rewriter.create<emitc::LoadOp>(loc, doubleType, sumVar).getResult();
    mlir::Value ne00Wide =
        rewriter.create<emitc::CastOp>(loc, doubleType, avlArg).getResult();
    mlir::Value meanDouble =
        rewriter.create<emitc::DivOp>(loc, doubleType, sumFinal, ne00Wide);
    mlir::Value mean =
        rewriter.create<emitc::CastOp>(loc, floatType, meanDouble).getResult();

    // float scale = 1.0f / sqrtf(mean + eps);  -- the add, sqrtf, and reciprocal
    // are ALL f32 (ggml's `1.0f/sqrtf(mean + eps)`; ops.cpp:3798). sqrtf is the
    // scalar libm call (call_opaque, the one sanctioned opaque seam) -- a true
    // IEEE correctly-rounded sqrt, NOT a hardware fast-rsqrt7. The reciprocal is
    // a separate f32 divide.
    mlir::Value sqrtVal = emitOpaqueCallBuilt(
        rewriter, loc, floatType, "sqrtf", opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value meanPlusEps =
              b.create<emitc::AddOp>(l, floatType, mean, eps);
          return {meanPlusEps};
        },
        llvm::StringRef("scale"));
    mlir::Value oneF =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
    mlir::Value scale =
        rewriter.create<emitc::DivOp>(loc, floatType, oneF, sqrtVal);

    // [FMT-PROP] FUSED-ACTIVATION-QUANTIZE detection: if the reduce core's fused
    // mul epilogue carries an OPTIONAL $quant_epilogue region (an
    // elementwise_quantize_q8_0_map brick), the WEIGHTED activation vz is quantized
    // to block_q8_0 IN REGISTER -- the f32 z[] intermediate is NEVER stored and the
    // downstream INDEPENDENT quantize_row_q8_0 pass (the f32 activation store + the
    // quantize reload, 2*n*4 bytes) is ELIDED. The fused-quant kernel is a per-BLOCK
    // loop (nb = n/32, vl = 32 in one e32m8 strip -- ggml's QK8_0 granularity the
    // reused amax/scale/narrow body needs), NOT the variable-length normalize strip
    // the plain / mul-only paths take below. The normalize + mul stay bare per-lane
    // vfmul (byte-exact at vl=32); the register-kept vz is bit-identical to a
    // store-then-reload, so the block_q8_0 is byte-exact to the non-fused
    // rms_norm->mul->quantize pipeline modulo ONLY the eliminated store/reload.
    tcrvrvv::ElementwiseMulMapOp fusedMul;
    rmsCore.getEpilogue().walk(
        [&](tcrvrvv::ElementwiseMulMapOp o) { fusedMul = o; });
    tcrvrvv::ElementwiseQuantizeQ80MapOp quantBrick;
    if (fusedMul)
      fusedMul.getQuantEpilogue().walk(
          [&](tcrvrvv::ElementwiseQuantizeQ80MapOp o) { quantBrick = o; });

    if (quantBrick) {
      mlir::Value weight = valueMap.lookup(fusedMul.getWeight());
      mlir::Value yq8 = valueMap.lookup(quantBrick.getOutput());
      if (!weight || !yq8)
        return rewriter.notifyMatchFailure(
            quantBrick, "fused quant epilogue ABI operand unmapped");

      llvm::StringRef mulOpName = fusedMul.getTCRVEmitCLowerableSourceOpName();
      llvm::StringRef mulRole = fusedMul.getTCRVEmitCLowerableSourceRole();
      llvm::StringRef qOpName = quantBrick.getTCRVEmitCLowerableSourceOpName();
      llvm::StringRef qRole = quantBrick.getTCRVEmitCLowerableSourceRole();

      // The AoS block-format facts (I4): qk=32 (block length / lanes),
      // block_stride=34, the fp16 d at byte 0, the 32 int8 qs at byte 2.
      int64_t qk = quantBrick.getQk();
      int64_t blockStride = quantBrick.getBlockStride();
      int64_t scaleOffset = quantBrick.getScaleByteOffset();
      int64_t quantOffset = quantBrick.getQuantByteOffset();

      mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
      mlir::Type weightPtrType = weight.getType();
      mlir::Type yq8PtrType = yq8.getType();

      auto qSizeLit = [&](int64_t v) -> mlir::Value {
        return rewriter.create<emitc::LiteralOp>(loc, sizeType,
                                                 std::to_string(v));
      };

      rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(qOpName, qRole));

      // size_t nb = n / 32;  (the AoS block count; n % 32 == 0 a ggml contract).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(qOpName, qRole, "block_count"));
      mlir::Value nb =
          rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, qSizeLit(qk));

      // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- ONE fused block loop
      // (normalize + mul + quantize per QK8_0 block).
      mlir::Value blkZero =
          rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
      mlir::Value blkOne = qSizeLit(1);
      auto blockFor = rewriter.create<emitc::ForOp>(loc, blkZero, nb, blkOne,
                                                    /*bodyBuilder=*/nullptr);
      mlir::Value ib = blockFor.getInductionVar();
      {
        mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
        rewriter.setInsertionPointToStart(blockFor.getBody());

        // size_t vl = 32;  (= QK8_0; all 32 block lanes in one e32m8 strip).
        mlir::Value vl = qSizeLit(qk);

        // const float *xb = (const float *)(x + ib*32);
        mlir::Value xOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, qSizeLit(qk));
        mlir::Value xbRaw =
            rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);
        mlir::Value xb =
            rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xbRaw)
                .getResult();

        // vfloat32m8_t vx = __riscv_vle32_v_f32m8(xb, vl);
        mlir::Value vx =
            emitOpaqueCall(rewriter, loc, f32m8Type, "__riscv_vle32_v_f32m8",
                           mlir::ValueRange{xb, vl}, opName, role);
        // vfloat32m8_t vy = __riscv_vfmul_vf_f32m8(vx, scale, vl);  normalize --
        // the register-kept normalized vector (no norm[] store).
        mlir::Value vy =
            emitOpaqueCall(rewriter, loc, f32m8Type, "__riscv_vfmul_vf_f32m8",
                           mlir::ValueRange{vx, scale, vl}, opName, role);

        // The mul epilogue: load the weight block, multiply vy in place.
        rewriter.create<emitc::VerbatimOp>(
            loc, routeSourceComment(mulOpName, mulRole));
        // const float *wb = (const float *)(w + ib*32);
        mlir::Value wbRaw =
            rewriter.create<emitc::AddOp>(loc, weightPtrType, weight, xOff);
        mlir::Value wb =
            rewriter.create<emitc::CastOp>(loc, constFloatPtrType, wbRaw)
                .getResult();
        // vfloat32m8_t vw = __riscv_vle32_v_f32m8(wb, vl);
        mlir::Value vw =
            emitOpaqueCall(rewriter, loc, f32m8Type, "__riscv_vle32_v_f32m8",
                           mlir::ValueRange{wb, vl}, mulOpName, mulRole);
        // vfloat32m8_t vz = __riscv_vfmul_vv_f32m8(vy, vw, vl);  the FUSED
        // multiply -- the register-kept vy flows STRAIGHT in, no z[] round-trip.
        mlir::Value vz =
            emitOpaqueCall(rewriter, loc, f32m8Type, "__riscv_vfmul_vv_f32m8",
                           mlir::ValueRange{vy, vw, vl}, mulOpName, mulRole);

        // The quant epilogue: the per-block amax/scale/narrow q8_0 body on the
        // register-kept vz -> block_q8_0 (yb = y_q8 + ib*34). The f32 z[] store is
        // GONE; the independent quantize_row_q8_0 reload is ELIDED.
        rewriter.create<emitc::VerbatimOp>(
            loc, routeSourceComment(qOpName, qRole));
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(qOpName, qRole, "y_block"));
        // uint8_t *yb = y_q8 + ib*34;  (the AoS block_q8_0 byte cursor).
        mlir::Value yOff = rewriter.create<emitc::MulOp>(
            loc, sizeType, ib, qSizeLit(blockStride));
        mlir::Value yb =
            rewriter.create<emitc::AddOp>(loc, yq8PtrType, yq8, yOff);
        emitQuantizeQ80BlockBody(rewriter, loc, vz, yb, vl, yq8PtrType, sizeType,
                                 scaleOffset, quantOffset, qOpName, qRole);
      }

      return mlir::success();
    }

    // The VECTORIZED normalize strip (step 4): y[i] = x[i] * scale. The NORMALIZE
    // strip LMUL is a bounded resource/scheduling fact (default m8, matching
    // ggml's ggml_vec_scale_f32 apply path). It is byte-exact at any anchor (a
    // bare per-lane vfmul_vf -- no FMA, no reduction). This is the SAME strip
    // machinery F1 (scale) emits, except two-buffer (x in, y out) instead of
    // in-place: byte-identical (both one f32 multiply per lane), avoiding ggml's
    // memcpy+in-place-scale.
    llvm::StringRef lmul = rmsCore.getStripLmul().value_or("m8");
    std::string f32VecTypeName = ("vfloat32" + lmul + "_t").str();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, f32VecTypeName);

    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    mlir::Value vlmax = emitOpaqueCall(rewriter, loc, sizeType, setvlCallee,
                                       mlir::ValueRange{avlArg}, opName, role);

    mlir::Value normZero =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto normFor = rewriter.create<emitc::ForOp>(loc, normZero, avlArg, vlmax,
                                                 /*bodyBuilder=*/nullptr);
    mlir::Value normIdx = normFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(normFor.getBody());

      // size_t vl = __riscv_vsetvl_e32m<L>(ne00 - i);
      mlir::Value bodyVL = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, setvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value remaining =
                b.create<emitc::SubOp>(l, sizeType, avlArg, normIdx);
            return {remaining};
          });

      // FUSED rms_norm->mul EPILOGUE (L2 region splice): if the reduce core
      // carries a tcrv_rvv.elementwise_mul_map brick in its $epilogue region, the
      // normalized vy is multiplied by the weight strip and stored to the fused
      // output z[] IN REGISTER -- the intermediate normalized row is NEVER stored
      // to memory and NEVER reloaded (the producer's normalize store + the
      // consumer's reload of norm[] are both elided; -2*n*4 bytes of DRAM
      // round-trip). BYTE-EXACT: the register-kept vy is bit-identical to a
      // store-then-reload of the same f32 vector, and `vy * w[i]` is a bare
      // per-lane fp32 multiply at the SAME LMUL anchor (no FMA -- no add follows;
      // no reduction). Detected up front so the UNFUSED path keeps the pre-fusion
      // op ORDER byte-identical (the y[] store pointer is materialized in its
      // original position, before the strip load).
      tcrvrvv::ElementwiseMulMapOp mulMap;
      rmsCore.getEpilogue().walk(
          [&](tcrvrvv::ElementwiseMulMapOp o) { mulMap = o; });

      // const float *xp = x + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, normIdx);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();

      // UNFUSED plain rms_norm: float *yp = y + i; materialized HERE (byte-exact
      // to the pre-fusion emit). The fused path never touches y[]; it computes
      // its z[] store pointer after the multiply instead.
      mlir::Value yStorePtr;
      if (!mulMap) {
        mlir::Value yPtr = rewriter.create<emitc::AddOp>(loc, outputPtrType,
                                                         outputBuf, normIdx);
        yStorePtr =
            rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();
      }

      // vfloat32m<L>_t vx = __riscv_vle32_v_f32m<L>(xp, vl);
      std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
      mlir::Value vx = emitOpaqueCall(rewriter, loc, f32VecType, loadCallee,
                                      mlir::ValueRange{xLoadPtr, bodyVL}, opName,
                                      role);

      // vfloat32m<L>_t vy = __riscv_vfmul_vf_f32m<L>(vx, scale, vl);  normalize.
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      mlir::Value vy = emitOpaqueCall(rewriter, loc, f32VecType, mulCallee,
                                      mlir::ValueRange{vx, scale, bodyVL}, opName,
                                      role);

      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");

      if (mulMap) {
        mlir::Value weight = valueMap.lookup(mulMap.getWeight());
        mlir::Value zOut = valueMap.lookup(mulMap.getOutput());
        if (!weight || !zOut)
          return rewriter.notifyMatchFailure(
              mulMap, "fused mul epilogue ABI operand unmapped");
        llvm::StringRef mulOpName = mulMap.getTCRVEmitCLowerableSourceOpName();
        llvm::StringRef mulRole = mulMap.getTCRVEmitCLowerableSourceRole();
        rewriter.create<emitc::VerbatimOp>(
            loc, routeSourceComment(mulOpName, mulRole));

        // const float *wp = w + i;  vfloat32m<L>_t vw = __riscv_vle32(wp, vl);
        mlir::Value wPtr = rewriter.create<emitc::AddOp>(loc, weight.getType(),
                                                         weight, normIdx);
        mlir::Value wLoadPtr =
            rewriter.create<emitc::CastOp>(loc, constFloatPtrType, wPtr)
                .getResult();
        mlir::Value vw = emitOpaqueCall(rewriter, loc, f32VecType, loadCallee,
                                        mlir::ValueRange{wLoadPtr, bodyVL},
                                        mulOpName, mulRole);

        // vfloat32m<L>_t vz = __riscv_vfmul_vv_f32m<L>(vy, vw, vl);  the fused
        // multiply -- vy flows straight in, no memory round-trip.
        std::string mulVVCallee = riscvIntrinsicName("vfmul", 32, lmul, "f32");
        mlir::Value vz = emitOpaqueCall(rewriter, loc, f32VecType, mulVVCallee,
                                        mlir::ValueRange{vy, vw, bodyVL},
                                        mulOpName, mulRole);

        // float *zp = z + i;  __riscv_vse32_v_f32m<L>(zp, vz, vl);
        mlir::Value zPtr = rewriter.create<emitc::AddOp>(loc, zOut.getType(),
                                                         zOut, normIdx);
        mlir::Value zStorePtr =
            rewriter.create<emitc::CastOp>(loc, floatPtrType, zPtr).getResult();
        emitOpaqueCallVoid(rewriter, loc, storeCallee,
                           mlir::ValueRange{zStorePtr, vz, bodyVL}, mulOpName,
                           mulRole);
      } else {
        // __riscv_vse32_v_f32m<L>(yp, vy, vl);  (plain rms_norm normalize store;
        // byte-identical to the pre-fusion emit).
        emitOpaqueCallVoid(rewriter, loc, storeCallee,
                           mlir::ValueRange{yStorePtr, vy, bodyVL}, opName, role);
      }
    }

    return mlir::success();
  }

mlir::Value VariantToEmitCFunc::emitGgmlVExpfM2(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Location loc, mlir::Value X,
                            mlir::Value bodyVL, mlir::Type sizeType,
                            llvm::StringRef opName, llvm::StringRef role) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m2_t");
    mlir::Type boolType = emitc::OpaqueType::get(ctx, "vbool16_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type u32ScalarType = emitc::OpaqueType::get(ctx, "uint32_t");

    auto vcall = [&](mlir::Type resTy, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resTy, callee, args, opName, role);
    };
    auto fimm = [&](llvm::StringRef tok) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, floatType, tok);
    };
    auto uimm = [&](llvm::StringRef tok) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, u32ScalarType, tok);
    };
    auto shiftImm = [&]() -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, "23");
    };

    // const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    mlir::Value r = vcall(f32VecType, "__riscv_vfmv_v_f_f32m2",
                          mlir::ValueRange{fimm("0x1.8p23f"), bodyVL});
    // const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    mlir::Value z = vcall(
        f32VecType, "__riscv_vfmacc_vf_f32m2",
        mlir::ValueRange{r, fimm("0x1.715476p+0f"), X, bodyVL});
    // const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    mlir::Value n = vcall(f32VecType, "__riscv_vfsub_vv_f32m2",
                          mlir::ValueRange{z, r, bodyVL});
    // const vfloat32m2_t b = vfnmsac(vfnmsac(x, 0x1.62e4p-1f, n), 0x1.7f7d1cp-20f, n);
    mlir::Value bInner = vcall(
        f32VecType, "__riscv_vfnmsac_vf_f32m2",
        mlir::ValueRange{X, fimm("0x1.62e4p-1f"), n, bodyVL});
    mlir::Value b = vcall(
        f32VecType, "__riscv_vfnmsac_vf_f32m2",
        mlir::ValueRange{bInner, fimm("0x1.7f7d1cp-20f"), n, bodyVL});
    // const vuint32m2_t e = __riscv_vsll_vx_u32m2(vreinterpret_v_f32m2_u32m2(z), 23, vl);
    mlir::Value zBits = vcall(u32VecType, "__riscv_vreinterpret_v_f32m2_u32m2",
                              mlir::ValueRange{z});
    mlir::Value e = vcall(u32VecType, "__riscv_vsll_vx_u32m2",
                          mlir::ValueRange{zBits, shiftImm(), bodyVL});
    // const vfloat32m2_t k = vreinterpret(vadd_vx_u32m2(e, 0x3f800000, vl));
    mlir::Value kBits = vcall(u32VecType, "__riscv_vadd_vx_u32m2",
                              mlir::ValueRange{e, uimm("0x3f800000"), bodyVL});
    mlir::Value k = vcall(f32VecType, "__riscv_vreinterpret_v_u32m2_f32m2",
                          mlir::ValueRange{kBits});
    // const vbool16_t c = vmfgt_vf(vfabs_v(n), 126.0f, vl);
    mlir::Value absN1 = vcall(f32VecType, "__riscv_vfabs_v_f32m2",
                              mlir::ValueRange{n, bodyVL});
    mlir::Value c = vcall(boolType, "__riscv_vmfgt_vf_f32m2_b16",
                          mlir::ValueRange{absN1, fimm("126.0f"), bodyVL});
    // const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    mlir::Value u = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                          mlir::ValueRange{b, b, bodyVL});
    // const vfloat32m2_t j = vfmacc_vv(
    //     vfmul_vf(b, 0x1.ffffecp-1f),
    //     vfmacc_vv(
    //         vfmacc_vf(vfmv_v_f(0x1.fffdb6p-2f), 0x1.555e66p-3f, b),
    //         vfmacc_vf(vfmv_v_f(0x1.573e2ep-5f), 0x1.0e4020p-7f, b),
    //         u),
    //     u);
    mlir::Value jOuterA = vcall(
        f32VecType, "__riscv_vfmul_vf_f32m2",
        mlir::ValueRange{b, fimm("0x1.ffffecp-1f"), bodyVL});
    mlir::Value jc0 = vcall(f32VecType, "__riscv_vfmv_v_f_f32m2",
                            mlir::ValueRange{fimm("0x1.fffdb6p-2f"), bodyVL});
    mlir::Value jInnerA = vcall(
        f32VecType, "__riscv_vfmacc_vf_f32m2",
        mlir::ValueRange{jc0, fimm("0x1.555e66p-3f"), b, bodyVL});
    mlir::Value jc1 = vcall(f32VecType, "__riscv_vfmv_v_f_f32m2",
                            mlir::ValueRange{fimm("0x1.573e2ep-5f"), bodyVL});
    mlir::Value jInnerB = vcall(
        f32VecType, "__riscv_vfmacc_vf_f32m2",
        mlir::ValueRange{jc1, fimm("0x1.0e4020p-7f"), b, bodyVL});
    mlir::Value jMid = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                             mlir::ValueRange{jInnerA, jInnerB, u, bodyVL});
    mlir::Value j = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                          mlir::ValueRange{jOuterA, jMid, u, bodyVL});

    // Slow path emitted UNCONDITIONALLY (the vcpop short-circuit is a pure perf
    // branch; the slow path's c-false/|n|<=192 lanes are bitwise-equal to the
    // fast path k + j*k). vec.h:1348-1359.
    // const vbool16_t dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    mlir::Value dm = vcall(boolType, "__riscv_vmfle_vf_f32m2_b16",
                           mlir::ValueRange{n, fimm("0.0f"), bodyVL});
    // const vuint32m2_t d = vmerge_vxm(vmv_v_x(0, vl), 0x82000000, dm, vl);
    mlir::Value dZero =
        vcall(u32VecType, "__riscv_vmv_v_x_u32m2",
              mlir::ValueRange{uimm("0"), bodyVL});
    mlir::Value d =
        vcall(u32VecType, "__riscv_vmerge_vxm_u32m2",
              mlir::ValueRange{dZero, uimm("0x82000000"), dm, bodyVL});
    // const vfloat32m2_t s1 = vreinterpret(vadd_vx_u32m2(d, 0x7f000000, vl));
    mlir::Value s1Bits = vcall(u32VecType, "__riscv_vadd_vx_u32m2",
                               mlir::ValueRange{d, uimm("0x7f000000"), bodyVL});
    mlir::Value s1 = vcall(f32VecType, "__riscv_vreinterpret_v_u32m2_f32m2",
                           mlir::ValueRange{s1Bits});
    // const vfloat32m2_t s2 = vreinterpret(vsub_vv_u32m2(e, d, vl));
    mlir::Value s2Bits = vcall(u32VecType, "__riscv_vsub_vv_u32m2",
                               mlir::ValueRange{e, d, bodyVL});
    mlir::Value s2 = vcall(f32VecType, "__riscv_vreinterpret_v_u32m2_f32m2",
                           mlir::ValueRange{s2Bits});
    // const vfloat32m2_t r1 = vmerge_vvm(
    //     vfmacc_vv(k, k, j, vl),
    //     vfmul_vv(vfmacc_vv(s2, s2, j, vl), s1, vl),
    //     c, vl);
    mlir::Value r1False = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                                mlir::ValueRange{k, k, j, bodyVL});
    mlir::Value r1TrueInner = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                                    mlir::ValueRange{s2, s2, j, bodyVL});
    mlir::Value r1True = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                               mlir::ValueRange{r1TrueInner, s1, bodyVL});
    mlir::Value r1 = vcall(f32VecType, "__riscv_vmerge_vvm_f32m2",
                           mlir::ValueRange{r1False, r1True, c, bodyVL});
    // return vmerge_vvm(r1, vfmul_vv(s1, s1, vl),
    //                   vmfgt_vf(vfabs_v(n), 192.0f, vl), vl);
    mlir::Value absN2 = vcall(f32VecType, "__riscv_vfabs_v_f32m2",
                              mlir::ValueRange{n, bodyVL});
    mlir::Value overMask = vcall(boolType, "__riscv_vmfgt_vf_f32m2_b16",
                                 mlir::ValueRange{absN2, fimm("192.0f"),
                                                  bodyVL});
    mlir::Value s1Sq = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                             mlir::ValueRange{s1, s1, bodyVL});
    return vcall(f32VecType, "__riscv_vmerge_vvm_f32m2",
                 mlir::ValueRange{r1, s1Sq, overMask, bodyVL});
  }

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseSiluMapStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::TypedElementwiseLoopBodyOp loopBody,
    tcrvrvv::ElementwiseSiluMapOp siluOp, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Anti-bypass (I7): the brick's strip_index MUST be the loop induction
    // variable (region arg 0), so the emit addresses x + i / y + i, not the
    // loop-invariant strip 0 (fail-closed otherwise). Same tie the scale map + the
    // flat/super-block per-block-source bricks enforce.
    mlir::Value stripIndex = loopBody.getBody().front().getArgument(0);
    if (siluOp.getStripIndex() != stripIndex)
      return rewriter.notifyMatchFailure(
          siluOp, "the elementwise_silu_map brick's strip_index must be the "
                  "loop induction variable (region arg 0)");

    // The ABI bases are sourced from the BRICK's operands (not the loop op), the
    // same anti-bypass convention the flat block-dot + scale-map bricks use.
    mlir::Value input = valueMap.lookup(siluOp.getInput());
    mlir::Value outputBuf = valueMap.lookup(siluOp.getOutput());
    if (!input || !outputBuf)
      return rewriter.notifyMatchFailure(siluOp, "silu ABI operand unmapped");

    llvm::StringRef opName = siluOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = siluOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = outputBuf.getType();

    // The strip loop is fixed at m2 (ggml's vsetvl_e32m2 path); the exp
    // polynomial's m2-tied mask/reinterpret types live in emitGgmlVExpfM2.
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // Small helpers: a structured emitc.call_opaque node, a literal float
    // immediate (rendered as the exact C hex-float token).
    auto vcall = [&](mlir::Type resTy, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resTy, callee, args, opName, role);
    };
    auto fimm = [&](llvm::StringRef tok) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, floatType, tok);
    };

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m2(n).
    mlir::Value vlmax =
        vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{avlArg});

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // size_t vl = __riscv_vsetvl_e32m2(n - i);
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{remaining});

      // const float *xp = x + i;  float *yp = y + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, inductionVar);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();
      mlir::Value yPtr = rewriter.create<emitc::AddOp>(loc, outputPtrType,
                                                       outputBuf, inductionVar);
      mlir::Value yStorePtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();

      // vfloat32m2_t vx = __riscv_vle32_v_f32m2(xp, vl);
      mlir::Value vx = vcall(f32VecType, "__riscv_vle32_v_f32m2",
                             mlir::ValueRange{xLoadPtr, bodyVL});

      // ===== ggml_v_silu_m2(vx, vl): neg -> ggml_v_expf_m2 -> +1 -> div =====
      // const vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
      mlir::Value negX = vcall(f32VecType, "__riscv_vfneg_v_f32m2",
                               mlir::ValueRange{vx, bodyVL});

      // exp(neg_x) via the SHARED node-for-node ggml_v_expf_m2 replication
      // (vec.h:1324) -- the IDENTICAL polynomial chain soft_max (F5b) consumes.
      mlir::Value expNegX =
          emitGgmlVExpfM2(rewriter, loc, negX, bodyVL, sizeType, opName, role);

      // const vfloat32m2_t one_plus = __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
      mlir::Value onePlus =
          vcall(f32VecType, "__riscv_vfadd_vf_f32m2",
                mlir::ValueRange{expNegX, fimm("1.0f"), bodyVL});
      // vfloat32m2_t vy = __riscv_vfdiv_vv_f32m2(x, one_plus, vl);
      mlir::Value vy = vcall(f32VecType, "__riscv_vfdiv_vv_f32m2",
                             mlir::ValueRange{vx, onePlus, bodyVL});

      // __riscv_vse32_v_f32m2(yp, vy, vl);
      emitOpaqueCallVoid(rewriter, loc, "__riscv_vse32_v_f32m2",
                         mlir::ValueRange{yStorePtr, vy, bodyVL}, opName, role);
    }

    return mlir::success();
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitElementwiseSoftMaxReduceStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The CONSTRUCTED soft_max reduce-model body, the exp-sum-reduce sibling of
    // the rms_norm Σx² reduce (emitElementwiseRmsNormReduceStrip) and of the
    // scale/silu map paths. The outer loop op owns the reduce shape
    // (reduce_map_model "reduce": a loop-carried f64m1 WIDENING accumulator region
    // arg + the yield that carries it back); this re-emit sources the WHOLE
    // soft_max ABI + the byte-exact fused exp-store-widening-reduce strip from the
    // region's reduce core brick (anti-bypass). BYTE-EXACT to the retired monolith
    // tcrv_rvv.ggml_vec_soft_max_f32 emit modulo ONLY the source-op provenance
    // token. Returns the f64 sum value (the dispatch wraps it in the function's
    // `return`).
    tcrvrvv::TypedElementwiseLoopBodyOp loopBody;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto lb = llvm::dyn_cast<tcrvrvv::TypedElementwiseLoopBodyOp>(op))
        loopBody = lb;
    }
    if (!loopBody)
      return rewriter.notifyMatchFailure(
          scope, "soft_max reduce body missing the typed elementwise loop op");
    tcrvrvv::ElementwiseSoftMaxReduceCoreOp softMaxCore;
    loopBody.getBody().walk(
        [&](tcrvrvv::ElementwiseSoftMaxReduceCoreOp o) { softMaxCore = o; });
    if (!softMaxCore)
      return rewriter.notifyMatchFailure(
          loopBody, "soft_max reduce-model body requires the soft_max reduce "
                    "core brick (elementwise_soft_max_reduce_core)");

    // Anti-bypass (I7): the brick's strip_index MUST be the loop induction
    // variable (region arg 0) and its acc MUST be the loop-carried accumulator
    // (region arg 1); the verifier pins both, checked here fail-closed too.
    mlir::Block &block = loopBody.getBody().front();
    if (block.getNumArguments() < 2 ||
        softMaxCore.getStripIndex() != block.getArgument(0) ||
        softMaxCore.getAcc() != block.getArgument(1))
      return rewriter.notifyMatchFailure(
          softMaxCore, "the soft_max reduce core brick's strip_index / acc must "
                       "be the loop induction variable / loop-carried "
                       "accumulator (region args 0 / 1)");

    mlir::Value outputBuf = valueMap.lookup(softMaxCore.getOutput());
    mlir::Value input = valueMap.lookup(softMaxCore.getInput());
    mlir::Value maxArg = valueMap.lookup(softMaxCore.getMax());
    if (!outputBuf || !input || !maxArg)
      return rewriter.notifyMatchFailure(softMaxCore,
                                         "soft_max ABI operand unmapped");

    llvm::StringRef opName = softMaxCore.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = softMaxCore.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = outputBuf.getType();

    // The strip loop is fixed at m2 (ggml's vsetvl_e32m2 path); the f64 reduce
    // accumulator is f64m1 (ggml's vfwredusum_vs_f32m2_f64m1 destination).
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type f64VecType = emitc::OpaqueType::get(ctx, "vfloat64m1_t");
    mlir::Type doubleType = emitc::OpaqueType::get(ctx, "double");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    auto vcall = [&](mlir::Type resTy, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resTy, callee, args, opName, role);
    };

    // vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);  -- the SCALAR f64
    // accumulator (ggml_float = double) carried across strips as a loop-carried
    // opaque-vector lvalue (emitc.for has no iter_args, exactly as F3 carries its
    // scalar double sum). ggml seeds it with vl=1.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("vsum", opName, role));
    auto vsumVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f64VecType),
        emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value zeroDouble =
        rewriter.create<emitc::LiteralOp>(loc, doubleType, "0.0");
    mlir::Value oneVL = rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    mlir::Value vsumInit = vcall(f64VecType, "__riscv_vfmv_v_f_f64m1",
                                 mlir::ValueRange{zeroDouble, oneVL});
    rewriter.create<emitc::AssignOp>(loc, vsumVar, vsumInit);

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m2(n).
    mlir::Value vlmax =
        vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{avlArg});

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // size_t vl = __riscv_vsetvl_e32m2(n - i);
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{remaining});

      // const float *xp = x + i;  float *yp = y + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, inductionVar);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();
      mlir::Value yPtr = rewriter.create<emitc::AddOp>(loc, outputPtrType,
                                                       outputBuf, inductionVar);
      mlir::Value yStorePtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();

      // vfloat32m2_t vx = __riscv_vle32_v_f32m2(xp, vl);
      mlir::Value vx = vcall(f32VecType, "__riscv_vle32_v_f32m2",
                             mlir::ValueRange{xLoadPtr, bodyVL});

      // vfloat32m2_t sub = __riscv_vfsub_vf_f32m2(vx, max, vl);  -- x[i]-max
      // (with x=-inf giving -inf, which the exp polynomial flushes to 0).
      mlir::Value sub = vcall(f32VecType, "__riscv_vfsub_vf_f32m2",
                              mlir::ValueRange{vx, maxArg, bodyVL});

      // vfloat32m2_t val = ggml_v_expf_m2(sub, vl) -- the SHARED node-for-node
      // exp polynomial (vec.h:1324), bit-identical to ggml's silu/soft_max.
      mlir::Value val =
          emitGgmlVExpfM2(rewriter, loc, sub, bodyVL, sizeType, opName, role);

      // __riscv_vse32_v_f32m2(yp, val, vl);   -- write y[i] = e^{x[i]-max}.
      emitOpaqueCallVoid(rewriter, loc, "__riscv_vse32_v_f32m2",
                         mlir::ValueRange{yStorePtr, val, bodyVL}, opName, role);

      // vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);  -- the WIDENING
      // f32->f64 reduce into the loop-carried f64m1 accumulator (ggml's EXACT
      // fold; the byte-exactness crux for the returned sum). Load the current
      // vsum, fold, assign back.
      mlir::Value vsumCur =
          rewriter.create<emitc::LoadOp>(loc, f64VecType, vsumVar).getResult();
      mlir::Value vsumNext =
          vcall(f64VecType, "__riscv_vfwredusum_vs_f32m2_f64m1",
                mlir::ValueRange{val, vsumCur, bodyVL});
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("vsum", opName, role));
      rewriter.create<emitc::AssignOp>(loc, vsumVar, vsumNext);
    }

    // return (double)__riscv_vfmv_f_s_f64m1_f64(vsum);  -- extract the f64 lane-0
    // scalar (ggml's `return (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum)`,
    // vec.cpp:592). The intrinsic already yields a `double`; the explicit cast
    // mirrors ggml's `(ggml_float)` and keeps the return-type contract.
    mlir::Value vsumFinal =
        rewriter.create<emitc::LoadOp>(loc, f64VecType, vsumVar).getResult();
    mlir::Value sumScalar = vcall(doubleType, "__riscv_vfmv_f_s_f64m1_f64",
                                  mlir::ValueRange{vsumFinal});
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "return"));
    mlir::Value sumReturn =
        rewriter.create<emitc::CastOp>(loc, doubleType, sumScalar).getResult();
    return sumReturn;
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlQuantizeRowQ80(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlQuantizeRowQ80Op quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<tcrvrvv::GgmlQuantizeRowQ80Op>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    mlir::Value input = valueMap.lookup(quantOp.getInput());
    mlir::Value output = valueMap.lookup(quantOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(quantOp,
                                         "quantize ABI operand unmapped");

    llvm::StringRef opName = quantOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = quantOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    // The block's 32 lanes ride in ONE e32m8 strip (ggml's `size_t vl = QK8_0`).
    // The wide-f32 load type is ggml's exact path type; the per-block
    // amax/scale/narrow body's remaining types live in emitQuantizeQ80BlockBody.
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");

    // The AoS block-format structural facts come straight off the typed attrs
    // (I4): qk=32 (block length / lanes), block_stride=34 (the AoS stride), the
    // fp16 d at byte 0, the 32 int8 qs at byte 2.
    int64_t qk = quantOp.getQk();
    int64_t blockStride = quantOp.getBlockStride();
    int64_t scaleOffset = quantOp.getScaleByteOffset();
    int64_t quantOffset = quantOp.getQuantByteOffset();

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto vcall = [&](mlir::Type resultType, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resultType, callee, args, opName,
                            role);
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / 32;  (ggml's `const int nb = k / QK8_0`). n % 32 == 0 is a
    // ggml contract (no tail), exactly as the q4_0 block-dot requires.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the AoS block loop.
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value one = sizeLit(1);
    auto blockFor = rewriter.create<emitc::ForOp>(loc, zero, nb, one,
                                                  /*bodyBuilder=*/nullptr);
    mlir::Value ib = blockFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(blockFor.getBody());

      // size_t vl = 32;  (ggml hard-pins `size_t vl = QK8_0` and relies on
      // Zvl128b => VLEN>=128 so the e32m8 strip holds all 32 lanes at once: ONE
      // vfabs, ONE vfredmax, ONE vfncvt per block).
      mlir::Value vl = sizeLit(qk);

      // const float *xb = x + ib*32;  (the f32 source block).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "x_block"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk));
      mlir::Value xbRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);
      mlir::Value xb =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xbRaw)
              .getResult();

      // uint8_t *yb = vy + ib*34;  (the AoS block_q8_0 byte cursor).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "y_block"));
      mlir::Value yOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(blockStride));
      mlir::Value yb =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, yOff);

      // v_x = __riscv_vle32_v_f32m8(xb, vl);
      mlir::Value vx =
          vcall(f32m8Type, "__riscv_vle32_v_f32m8", mlir::ValueRange{xb, vl});

      // The SHARED per-block amax/scale/narrow q8_0 body (riscv/quants.c:47-65),
      // consuming the f32 block LOADED from x[]. The [FMT-PROP] fused
      // rms_norm->mul->quantize epilogue reuses this SAME body on a register-kept
      // weighted vector `vz` instead (no f32 store/reload).
      emitQuantizeQ80BlockBody(rewriter, loc, vx, yb, vl, outputPtrType, sizeType,
                               scaleOffset, quantOffset, opName, role);
    }

    return mlir::success();
  }

void VariantToEmitCFunc::emitQuantizeQ80BlockBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value vBlock, mlir::Value yb, mlir::Value vl, mlir::Type outputPtrType,
    mlir::Type sizeType, int64_t scaleOffset, int64_t quantOffset,
    llvm::StringRef opName, llvm::StringRef role) const {
    // ggml's per-block block_q8_0 amax/scale/narrow body (riscv/quants.c:47-65) as
    // fully STRUCTURED emitc nodes, over an ALREADY-COMPUTED f32m8 block vector
    // `vBlock` (the 32 QK8_0 lanes in one e32m8 strip). This is the SHARED
    // quantize core: the standalone f32->q8_0 activation quantizer feeds it the
    // f32 block loaded from x[]; the [FMT-PROP] fused rms_norm->mul->quantize
    // epilogue feeds it the register-kept WEIGHTED vector vz. BYTE-EXACT to ggml's
    // EXACT RVV method (vfncvt = rne + native _Float16 cast).
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type half16Type = emitc::OpaqueType::get(ctx, "_Float16");
    mlir::Type half16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "_Float16"));
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "int8_t"));
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
    mlir::Type f32m1Type = emitc::OpaqueType::get(ctx, "vfloat32m1_t");
    mlir::Type i16m4Type = emitc::OpaqueType::get(ctx, "vint16m4_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto vcall = [&](mlir::Type resultType, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resultType, callee, args, opName,
                            role);
    };

    // The amax reduction (ggml riscv/quants.c:47-50): vfabs -> vfredmax seeded
    // with a 0.0f f32m1 -> extract lane 0. The intrinsic callees are HARD-CODED
    // (ggml's exact spellings) -- never synthesized.
    mlir::Value vabs =
        vcall(f32m8Type, "__riscv_vfabs_v_f32m8", mlir::ValueRange{vBlock, vl});
    mlir::Value zeroF =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    mlir::Value redSeed = vcall(f32m1Type, "__riscv_vfmv_v_f_f32m1",
                                mlir::ValueRange{zeroF, vl});
    mlir::Value vmax = vcall(f32m1Type, "__riscv_vfredmax_vs_f32m8_f32m1",
                             mlir::ValueRange{vabs, redSeed, vl});
    mlir::Value amax = vcall(floatType, "__riscv_vfmv_f_s_f32m1_f32",
                             mlir::ValueRange{vmax});

    // float d = amax / 127.0f;  (ggml's `amax / ((1 << 7) - 1)`; the divisor is
    // the f32 literal 127.0f so the divide is a single f32 round).
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
    mlir::Value c127 =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "127.0f");
    mlir::Value d = rewriter.create<emitc::DivOp>(loc, floatType, amax, c127);

    // float id = 0.0f; if (d != 0.0f) { id = 1.0f / d; }  -- the load-bearing
    // `id = d ? 1.0f/d : 0.0f` conditional, STRUCTURED (emitc.cmp + emitc.if, NOT
    // a raw string). The all-zero block (amax=0 => d=0) takes the else and keeps
    // id=0, so every q=0 (a bare 1/d would give inf, then 0*inf=NaN).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("id", opName, role));
    auto idVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, idVar, rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
    mlir::Value dNonZero = rewriter.create<emitc::CmpOp>(
        loc, boolType, emitc::CmpPredicate::ne, d, zeroF);
    auto idIf = rewriter.create<emitc::IfOp>(loc, dNonZero,
                                             /*addThenBlock=*/true,
                                             /*addElseBlock=*/false);
    {
      mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
      rewriter.setInsertionPointToStart(&idIf.getThenRegion().front());
      mlir::Value oneF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
      mlir::Value recip = rewriter.create<emitc::DivOp>(loc, floatType, oneF, d);
      rewriter.create<emitc::AssignOp>(loc, idVar, recip);
      rewriter.create<emitc::YieldOp>(loc);
    }
    mlir::Value id =
        rewriter.create<emitc::LoadOp>(loc, floatType, idVar).getResult();

    // *(_Float16 *)(yb + 0) = (_Float16)d;  -- the fp16 d store. The board is
    // __riscv_zfhmin, so GGML_CPU_FP32_TO_FP16(d) is the native (_Float16)d cast
    // (fcvt.h.s, rne). STRUCTURED: cast the byte cursor to _Float16 *, subscript
    // [0] (an lvalue), cast d to _Float16, assign.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "fp16_d_store"));
    mlir::Value dPtrRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                                        sizeLit(scaleOffset));
    auto dPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
        rewriter.create<emitc::CastOp>(loc, half16PtrType, dPtrRaw).getResult());
    mlir::Value dIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp dSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, dPtr, dIndex);
    mlir::Value dHalf =
        rewriter.create<emitc::CastOp>(loc, half16Type, d).getResult();
    rewriter.create<emitc::AssignOp>(loc, dSubscript.getResult(), dHalf);

    // x0 = __riscv_vfmul_vf_f32m8(vBlock, id, vl);  -- scale every lane by id.
    mlir::Value x0 = vcall(f32m8Type, "__riscv_vfmul_vf_f32m8",
                           mlir::ValueRange{vBlock, id, vl});

    // The NARROWING CONVERT (ggml riscv/quants.c:60-61): f32 -> i16 (the rounding
    // crux: vfncvt_x_f_w_i16m4 = dynamic frm = round-to-nearest-EVEN), then
    // i16 -> i8 truncate (vncvt). Both callees are ggml's exact spellings.
    mlir::Value vi = vcall(i16m4Type, "__riscv_vfncvt_x_f_w_i16m4",
                           mlir::ValueRange{x0, vl});
    mlir::Value vs =
        vcall(i8m2Type, "__riscv_vncvt_x_x_w_i8m2", mlir::ValueRange{vi, vl});

    // __riscv_vse8_v_i8m2(yb + 2, vs, vl);  -- store the 32 int8 qs.
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "qs_store"));
    mlir::Value qsPtrRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                                         sizeLit(quantOffset));
    mlir::Value qsPtr =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, qsPtrRaw).getResult();
    emitOpaqueCallVoid(rewriter, loc, "__riscv_vse8_v_i8m2",
                       mlir::ValueRange{qsPtr, vs, vl}, opName, role);
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlQuantizeRowQ81(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The q8_1 SIBLING of emitGgmlQuantizeRowQ80: the SAME per-32-block amax
    // reduction + scale + f32->i16->i8 narrow + AoS store, PLUS the extra
    // vwredsum integer block sum stored as the fp16 block_q8_1.s (ggml
    // riscv/quants.c). DISPATCH-WIRED ([L-6] wiring != construction).
    tcrvrvv::GgmlQuantizeRowQ81Op quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<tcrvrvv::GgmlQuantizeRowQ81Op>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    mlir::Value input = valueMap.lookup(quantOp.getInput());
    mlir::Value output = valueMap.lookup(quantOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(quantOp,
                                         "quantize ABI operand unmapped");

    llvm::StringRef opName = quantOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = quantOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type half16Type = emitc::OpaqueType::get(ctx, "_Float16");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type half16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "_Float16"));
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "int8_t"));
    // The block's 32 lanes ride in ONE e32m8 strip (ggml's `size_t vl = QK8_1`);
    // the wide f32 / m1 reduce / i16m4 / i8m2 types are ggml's exact path types;
    // the block sum widens the i8m2 quants into an i16m1 accumulator.
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
    mlir::Type f32m1Type = emitc::OpaqueType::get(ctx, "vfloat32m1_t");
    mlir::Type i16m4Type = emitc::OpaqueType::get(ctx, "vint16m4_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");

    // The AoS block-format structural facts come straight off the typed attrs
    // (I4): qk=32 (block length / lanes), block_stride=36 (the AoS stride), the
    // fp16 d at byte 0, the fp16 s at byte 2, the 32 int8 qs at byte 4.
    int64_t qk = quantOp.getQk();
    int64_t blockStride = quantOp.getBlockStride();
    int64_t scaleOffset = quantOp.getScaleByteOffset();
    int64_t sumOffset = quantOp.getSumByteOffset();
    int64_t quantOffset = quantOp.getQuantByteOffset();

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto vcall = [&](mlir::Type resultType, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resultType, callee, args, opName,
                            role);
    };
    // Store a scalar `value` as a native (_Float16) at yb + `byteOffset` (the
    // board is __riscv_zfhmin, so the fp16 store is a structural _Float16 cast +
    // subscript assign; ggml's GGML_CPU_FP32_TO_FP16).
    auto storeHalf = [&](mlir::Value yb, int64_t byteOffset, mlir::Value value) {
      mlir::Value ptrRaw = rewriter.create<emitc::AddOp>(
          loc, outputPtrType, yb, sizeLit(byteOffset));
      auto ptr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, half16PtrType, ptrRaw)
              .getResult());
      mlir::Value idx =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      emitc::SubscriptOp sub =
          rewriter.create<emitc::SubscriptOp>(loc, ptr, idx);
      mlir::Value half =
          rewriter.create<emitc::CastOp>(loc, half16Type, value).getResult();
      rewriter.create<emitc::AssignOp>(loc, sub.getResult(), half);
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / 32;  (ggml's `const int nb = k / QK8_1`).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the AoS block loop.
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value one = sizeLit(1);
    auto blockFor = rewriter.create<emitc::ForOp>(loc, zero, nb, one,
                                                  /*bodyBuilder=*/nullptr);
    mlir::Value ib = blockFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(blockFor.getBody());

      // size_t vl = 32;  (ggml hard-pins `size_t vl = QK8_1`).
      mlir::Value vl = sizeLit(qk);

      // const float *xb = x + ib*32;  (the f32 source block).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "x_block"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk));
      mlir::Value xbRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);
      mlir::Value xb =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xbRaw)
              .getResult();

      // uint8_t *yb = vy + ib*36;  (the AoS block_q8_1 byte cursor).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "y_block"));
      mlir::Value yOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(blockStride));
      mlir::Value yb =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, yOff);

      // v_x = __riscv_vle32_v_f32m8(xb, vl);
      mlir::Value vx =
          vcall(f32m8Type, "__riscv_vle32_v_f32m8", mlir::ValueRange{xb, vl});

      // amax = vfmv_f_s(vfredmax(vfabs(v_x), 0.0f)); (ggml riscv/quants.c).
      mlir::Value vabs =
          vcall(f32m8Type, "__riscv_vfabs_v_f32m8", mlir::ValueRange{vx, vl});
      mlir::Value zeroF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
      mlir::Value redSeed = vcall(f32m1Type, "__riscv_vfmv_v_f_f32m1",
                                  mlir::ValueRange{zeroF, vl});
      mlir::Value vmax =
          vcall(f32m1Type, "__riscv_vfredmax_vs_f32m8_f32m1",
                mlir::ValueRange{vabs, redSeed, vl});
      mlir::Value amax = vcall(floatType, "__riscv_vfmv_f_s_f32m1_f32",
                               mlir::ValueRange{vmax});

      // float d = amax / 127.0f;
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
      mlir::Value c127 =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "127.0f");
      mlir::Value d = rewriter.create<emitc::DivOp>(loc, floatType, amax, c127);

      // float id = 0.0f; if (d != 0.0f) { id = 1.0f / d; }  -- the load-bearing
      // `id = d ? 1.0f/d : 0.0f` conditional, STRUCTURED (emitc.cmp + emitc.if).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("id", opName, role));
      auto idVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(floatType),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, idVar,
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
      mlir::Value dNonZero = rewriter.create<emitc::CmpOp>(
          loc, boolType, emitc::CmpPredicate::ne, d, zeroF);
      auto idIf = rewriter.create<emitc::IfOp>(loc, dNonZero,
                                               /*addThenBlock=*/true,
                                               /*addElseBlock=*/false);
      {
        mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
        rewriter.setInsertionPointToStart(&idIf.getThenRegion().front());
        mlir::Value oneF =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
        mlir::Value recip =
            rewriter.create<emitc::DivOp>(loc, floatType, oneF, d);
        rewriter.create<emitc::AssignOp>(loc, idVar, recip);
        rewriter.create<emitc::YieldOp>(loc);
      }
      mlir::Value id =
          rewriter.create<emitc::LoadOp>(loc, floatType, idVar).getResult();

      // *(_Float16 *)(yb + 0) = (_Float16)d;  -- the fp16 d store.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp16_d_store"));
      storeHalf(yb, scaleOffset, d);

      // x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);  -- scale every lane by id.
      mlir::Value x0 = vcall(f32m8Type, "__riscv_vfmul_vf_f32m8",
                             mlir::ValueRange{vx, id, vl});

      // The NARROWING CONVERT: f32 -> i16 (vfncvt = round-to-nearest-EVEN) then
      // i16 -> i8 truncate (vncvt).
      mlir::Value vi = vcall(i16m4Type, "__riscv_vfncvt_x_f_w_i16m4",
                             mlir::ValueRange{x0, vl});
      mlir::Value vs = vcall(i8m2Type, "__riscv_vncvt_x_x_w_i8m2",
                             mlir::ValueRange{vi, vl});

      // __riscv_vse8_v_i8m2(yb + 4, vs, vl);  -- store the 32 int8 qs.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qs_store"));
      mlir::Value qsPtrRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                        sizeLit(quantOffset));
      mlir::Value qsPtr =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, qsPtrRaw).getResult();
      emitOpaqueCallVoid(rewriter, loc, "__riscv_vse8_v_i8m2",
                         mlir::ValueRange{qsPtr, vs, vl}, opName, role);

      // The block sum (ggml riscv/quants.c): a WIDENING integer reduction of the
      // int8 quants into an i16m1 accumulator seeded 0, then the scalar extract,
      // then s = (_Float16)(sum * d) at AoS byte 2.
      //   vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
      //   vint16m1_t vwrs = __riscv_vwredsum_vs_i8m2_i16m1(vs, tmp2, vl);
      //   int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "block_sum"));
      mlir::Value intZero =
          rewriter.create<emitc::LiteralOp>(loc, intType, "0");
      mlir::Value sumSeed = vcall(i16m1Type, "__riscv_vmv_v_x_i16m1",
                                  mlir::ValueRange{intZero, vl});
      mlir::Value vwrs = vcall(i16m1Type, "__riscv_vwredsum_vs_i8m2_i16m1",
                               mlir::ValueRange{vs, sumSeed, vl});
      mlir::Value sum = vcall(intType, "__riscv_vmv_x_s_i16m1_i16",
                              mlir::ValueRange{vwrs});

      // *(_Float16 *)(yb + 2) = (_Float16)(sum * d);  -- the block_q8_1.s store.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp16_s_store"));
      mlir::Value sumD = rewriter.create<emitc::MulOp>(loc, floatType, sum, d);
      storeHalf(yb, sumOffset, sumD);
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlQuantizeRowQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The heaviest quantizer (ggml quantize_row_q8_K RVV path, riscv/quants.c):
    // a QK_K=256 super-block loop -- a min/max reduction, a symmetric iscale, the
    // vfcvt/vnclip RNE narrow, the FLOAT d store, the per-16 vwredsum bsums, and
    // the zero-block memset special case. DISPATCH-WIRED ([L-6] wiring !=
    // construction).
    tcrvrvv::GgmlQuantizeRowQ8KOp quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<tcrvrvv::GgmlQuantizeRowQ8KOp>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    mlir::Value input = valueMap.lookup(quantOp.getInput());
    mlir::Value output = valueMap.lookup(quantOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(quantOp,
                                         "quantize ABI operand unmapped");

    llvm::StringRef opName = quantOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = quantOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type indexType = rewriter.getIndexType();
    mlir::Type i16Type = emitc::OpaqueType::get(ctx, "int16_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type frmType = emitc::OpaqueType::get(ctx, "unsigned int");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "int8_t"));
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "int16_t"));
    // ggml's exact path types: the e32m8 min/max/quant strips, the m1 reduce
    // seeds, the i32m8 fcvt, the i16m4/i8m2 narrow, and the i8m1/i16m1 per-16
    // bsums chunk/accumulator.
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
    mlir::Type f32m1Type = emitc::OpaqueType::get(ctx, "vfloat32m1_t");
    mlir::Type i32m8Type = emitc::OpaqueType::get(ctx, "vint32m8_t");
    mlir::Type i16m4Type = emitc::OpaqueType::get(ctx, "vint16m4_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8m1Type = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");

    // The AoS block-format structural facts come straight off the typed attrs
    // (I4): qk=256 (QK_K), block_stride=292, the FLOAT d at byte 0, the 256 int8
    // qs at byte 4, the 16 int16 bsums at byte 260.
    int64_t qk = quantOp.getQk();
    int64_t blockStride = quantOp.getBlockStride();
    int64_t scaleOffset = quantOp.getScaleByteOffset();
    int64_t quantOffset = quantOp.getQuantByteOffset();
    int64_t bsumsOffset = quantOp.getBsumsByteOffset();

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto vcall = [&](mlir::Type resultType, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, resultType, callee, args, opName,
                            role);
    };
    auto fcall = [&](llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, floatType, callee, args, opName,
                            role);
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / 256;  (ggml's `size_t nb = k / QK_K`).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // const size_t vlmax = __riscv_vsetvlmax_e32m8();  -- the shared e32m8 vlmax
    // ggml folds the min/max init + reduction over.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "vlmax"));
    mlir::Value vlmax = emitOpaqueCall(rewriter, loc, sizeType,
                                       "__riscv_vsetvlmax_e32m8",
                                       mlir::ValueRange{}, opName, role);

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the super-block loop.
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value one = sizeLit(1);
    auto blockFor = rewriter.create<emitc::ForOp>(loc, zero, nb, one,
                                                  /*bodyBuilder=*/nullptr);
    mlir::Value ib = blockFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(blockFor.getBody());

      // const float *xb = x + ib*256;  uint8_t *yb = vy + ib*292;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "x_block"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk));
      mlir::Value xbRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);
      mlir::Value xb =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xbRaw)
              .getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "y_block"));
      mlir::Value yOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(blockStride));
      mlir::Value yb =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, yOff);

      // 1. min/max over the 256-lane super-block: seed max/min m8 vectors to
      // -/+inf, fold over an e32m8 strip loop (vfmax_vv/vfmin_vv), then reduce to
      // scalars (vfredmax/vfredmin seeded -/+inf).
      mlir::Value negInf =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "-__builtin_inff()");
      mlir::Value posInf =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "__builtin_inff()");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "minmax_init"));
      auto maxVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(f32m8Type),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, maxVar,
          vcall(f32m8Type, "__riscv_vfmv_v_f_f32m8",
                mlir::ValueRange{negInf, vlmax}));
      auto minVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(f32m8Type),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, minVar,
          vcall(f32m8Type, "__riscv_vfmv_v_f_f32m8",
                mlir::ValueRange{posInf, vlmax}));

      // for (size_t off = 0; off < 256; off += vlmax) {
      //   size_t vl = __riscv_vsetvl_e32m8(256 - off);
      //   v_curr = vle32(xb + off, vl);
      //   max_v = vfmax_vv(max_v, v_curr, vl); min_v = vfmin_vv(min_v, v_curr, vl);
      // }  (byte-exact to ggml's `while (rem>0)` strip loop: same vl sequence,
      //     same data, and min/max is order-independent.)
      auto minmaxFor = rewriter.create<emitc::ForOp>(loc, zero, sizeLit(qk),
                                                     vlmax,
                                                     /*bodyBuilder=*/nullptr);
      mlir::Value mmOff = minmaxFor.getInductionVar();
      {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(minmaxFor.getBody());
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, "__riscv_vsetvl_e32m8", opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value rem =
                  b.create<emitc::SubOp>(l, sizeType, sizeLit(qk), mmOff);
              return {rem};
            });
        mlir::Value pRaw =
            rewriter.create<emitc::AddOp>(loc, constFloatPtrType, xb, mmOff);
        mlir::Value vCurr = vcall(f32m8Type, "__riscv_vle32_v_f32m8",
                                  mlir::ValueRange{pRaw, vl});
        mlir::Value maxCur =
            rewriter.create<emitc::LoadOp>(loc, f32m8Type, maxVar).getResult();
        mlir::Value minCur =
            rewriter.create<emitc::LoadOp>(loc, f32m8Type, minVar).getResult();
        rewriter.create<emitc::AssignOp>(
            loc, maxVar,
            vcall(f32m8Type, "__riscv_vfmax_vv_f32m8",
                  mlir::ValueRange{maxCur, vCurr, vl}));
        rewriter.create<emitc::AssignOp>(
            loc, minVar,
            vcall(f32m8Type, "__riscv_vfmin_vv_f32m8",
                  mlir::ValueRange{minCur, vCurr, vl}));
      }

      // Reduce to scalars: seed m1 lane0 to -/+inf, vfredmax/vfredmin, extract.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "minmax_reduce"));
      mlir::Value oneLit = sizeLit(1);
      mlir::Value initMax = vcall(f32m1Type, "__riscv_vfmv_s_f_f32m1",
                                  mlir::ValueRange{negInf, oneLit});
      mlir::Value initMin = vcall(f32m1Type, "__riscv_vfmv_s_f_f32m1",
                                  mlir::ValueRange{posInf, oneLit});
      mlir::Value maxAll =
          rewriter.create<emitc::LoadOp>(loc, f32m8Type, maxVar).getResult();
      mlir::Value minAll =
          rewriter.create<emitc::LoadOp>(loc, f32m8Type, minVar).getResult();
      mlir::Value vScalarMax =
          vcall(f32m1Type, "__riscv_vfredmax_vs_f32m8_f32m1",
                mlir::ValueRange{maxAll, initMax, vlmax});
      mlir::Value vScalarMin =
          vcall(f32m1Type, "__riscv_vfredmin_vs_f32m8_f32m1",
                mlir::ValueRange{minAll, initMin, vlmax});
      mlir::Value maxVal = vcall(floatType, "__riscv_vfmv_f_s_f32m1_f32",
                                 mlir::ValueRange{vScalarMax});
      mlir::Value minVal = vcall(floatType, "__riscv_vfmv_f_s_f32m1_f32",
                                 mlir::ValueRange{vScalarMin});

      // float amax = fabsf(max_val) > fabsf(min_val) ? fabsf(max_val)
      //                                              : fabsf(min_val);  and the
      // symmetric selector max_val>min_val branch reuse fabsf (the scalar seam).
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "amax"));
      mlir::Value fabsMax = fcall("fabsf", mlir::ValueRange{maxVal});
      mlir::Value fabsMin = fcall("fabsf", mlir::ValueRange{minVal});
      mlir::Value maxAbsGt = rewriter.create<emitc::CmpOp>(
          loc, boolType, emitc::CmpPredicate::gt, fabsMax, fabsMin);

      // amax = maxAbsGt ? fabsMax : fabsMin;   iscale_den = maxAbsGt ? max : min;
      // -- both are STRUCTURED emitc.variable + emitc.if (the ternary lowered).
      auto amaxVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(floatType),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(loc, amaxVar, fabsMin);
      auto denVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(floatType),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(loc, denVar, minVal);
      auto amaxIf = rewriter.create<emitc::IfOp>(loc, maxAbsGt,
                                                 /*addThenBlock=*/true,
                                                 /*addElseBlock=*/false);
      {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(&amaxIf.getThenRegion().front());
        rewriter.create<emitc::AssignOp>(loc, amaxVar, fabsMax);
        rewriter.create<emitc::AssignOp>(loc, denVar, maxVal);
        rewriter.create<emitc::YieldOp>(loc);
      }
      mlir::Value amax =
          rewriter.create<emitc::LoadOp>(loc, floatType, amaxVar).getResult();
      mlir::Value den =
          rewriter.create<emitc::LoadOp>(loc, floatType, denVar).getResult();

      // if (amax == 0.0f) { d=0; memset(qs,0,256); memset(bsums,0,32); }
      // else { d = 1/iscale; quantize + bsums }   -- ggml's `continue` lowered to
      // a STRUCTURED emitc.if/else (byte-exact: the else IS the non-zero path).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "zero_block_guard"));
      mlir::Value zeroF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
      mlir::Value amaxZero = rewriter.create<emitc::CmpOp>(
          loc, boolType, emitc::CmpPredicate::eq, amax, zeroF);
      auto zeroIf = rewriter.create<emitc::IfOp>(loc, amaxZero,
                                                 /*addThenBlock=*/true,
                                                 /*addElseBlock=*/true);

      // Common: cast yb+4 to int8_t *qs, yb+260 to int16_t *bsums, yb+0 to float
      // *d for the stores below.
      auto qsBasePtr = [&]() -> mlir::Value {
        mlir::Value raw = rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                                        sizeLit(quantOffset));
        return rewriter.create<emitc::CastOp>(loc, i8PtrType, raw).getResult();
      };
      auto bsumsBasePtr = [&]() -> mlir::TypedValue<emitc::PointerType> {
        mlir::Value raw = rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                                        sizeLit(bsumsOffset));
        return llvm::cast<mlir::TypedValue<emitc::PointerType>>(
            rewriter.create<emitc::CastOp>(loc, i16PtrType, raw).getResult());
      };
      // bsums[idx] = value;  emitted via int16_t* pointer arithmetic (base + idx,
      // element stride) + subscript[0] -- the same pointer-arithmetic idiom the
      // block cursors use, avoiding a computed-index subscript.
      auto storeBsum = [&](mlir::Value idx, mlir::Value value) {
        mlir::Value elemPtr =
            rewriter.create<emitc::AddOp>(loc, i16PtrType, bsumsBasePtr(), idx);
        auto ptr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(elemPtr);
        mlir::Value z =
            rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
        emitc::SubscriptOp sub =
            rewriter.create<emitc::SubscriptOp>(loc, ptr, z);
        rewriter.create<emitc::AssignOp>(loc, sub.getResult(), value);
      };
      auto storeFloatD = [&](mlir::Value value) {
        mlir::Value raw = rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                                        sizeLit(scaleOffset));
        auto ptr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
            rewriter.create<emitc::CastOp>(loc, floatPtrType, raw).getResult());
        mlir::Value idx =
            rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
        emitc::SubscriptOp sub =
            rewriter.create<emitc::SubscriptOp>(loc, ptr, idx);
        rewriter.create<emitc::AssignOp>(loc, sub.getResult(), value);
      };

      // The zero path: y_block->d = 0.0f; memset(qs,0,256); memset(bsums,0,32);
      {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(&zeroIf.getThenRegion().front());
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "zero_block"));
        storeFloatD(rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
        mlir::Value memZero =
            rewriter.create<emitc::LiteralOp>(loc, intType, "0");
        emitOpaqueCallVoid(
            rewriter, loc, "memset",
            mlir::ValueRange{qsBasePtr(), memZero, sizeLit(qk)}, opName, role);
        emitOpaqueCallVoid(rewriter, loc, "memset",
                           mlir::ValueRange{bsumsBasePtr(), memZero,
                                            sizeLit(blockStride - bsumsOffset)},
                           opName, role);
        rewriter.create<emitc::YieldOp>(loc);
      }

      // The non-zero path: iscale = -127 / den; d = 1/iscale; quantize + bsums.
      {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(&zeroIf.getElseRegion().front());
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "iscale"));
        mlir::Value negC127 =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "-127.f");
        mlir::Value iscale =
            rewriter.create<emitc::DivOp>(loc, floatType, negC127, den);
        mlir::Value oneF =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
        mlir::Value d =
            rewriter.create<emitc::DivOp>(loc, floatType, oneF, iscale);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "float_d_store"));
        storeFloatD(d);

        // The RNE rounding-mode / saturation-mode enum operands.
        mlir::Value frmRne = rewriter.create<emitc::LiteralOp>(
            loc, frmType, "__RISCV_FRM_RNE");
        mlir::Value vxrmRne = rewriter.create<emitc::LiteralOp>(
            loc, frmType, "__RISCV_VXRM_RNE");
        mlir::Value i16zeroChunk =
            rewriter.create<emitc::LiteralOp>(loc, intType, "0");
        // vint16m1_t v_zero_sum = __riscv_vmv_v_x_i16m1(0, 1);
        mlir::Value zeroSum = vcall(i16m1Type, "__riscv_vmv_v_x_i16m1",
                                    mlir::ValueRange{i16zeroChunk, oneLit});
        mlir::Value nclipShift =
            rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");

        // for (size_t off = 0; off < 256; off += vlmax) { quantize strip + bsums }
        auto quantFor = rewriter.create<emitc::ForOp>(loc, zero, sizeLit(qk),
                                                      vlmax,
                                                      /*bodyBuilder=*/nullptr);
        mlir::Value qOff = quantFor.getInductionVar();
        {
          mlir::OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(quantFor.getBody());
          mlir::Value vl = emitOpaqueCallBuilt(
              rewriter, loc, sizeType, "__riscv_vsetvl_e32m8", opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value rem =
                    b.create<emitc::SubOp>(l, sizeType, sizeLit(qk), qOff);
                return {rem};
              });
          // v_f = vle32(xb + off, vl); v_f = vfmul_vf(v_f, iscale, vl);
          mlir::Value pRaw =
              rewriter.create<emitc::AddOp>(loc, constFloatPtrType, xb, qOff);
          mlir::Value vF = vcall(f32m8Type, "__riscv_vle32_v_f32m8",
                                 mlir::ValueRange{pRaw, vl});
          mlir::Value vScaled = vcall(f32m8Type, "__riscv_vfmul_vf_f32m8",
                                      mlir::ValueRange{vF, iscale, vl});
          // v_i32 = vfcvt_x_f_v_i32m8_rm(v_f, RNE, vl);
          // v_i16 = vnclip_wx_i16m4(v_i32, 0, RNE, vl);
          // v_q   = vnclip_wx_i8m2(v_i16, 0, RNE, vl);
          mlir::Value vI32 =
              vcall(i32m8Type, "__riscv_vfcvt_x_f_v_i32m8_rm",
                    mlir::ValueRange{vScaled, frmRne, vl});
          mlir::Value vI16 =
              vcall(i16m4Type, "__riscv_vnclip_wx_i16m4",
                    mlir::ValueRange{vI32, nclipShift, vxrmRne, vl});
          mlir::Value vQ =
              vcall(i8m2Type, "__riscv_vnclip_wx_i8m2",
                    mlir::ValueRange{vI16, nclipShift, vxrmRne, vl});
          // __riscv_vse8_v_i8m2(qs + off, v_q, vl);
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "qs_store"));
          mlir::Value qsOffPtr = rewriter.create<emitc::AddOp>(
              loc, i8PtrType, qsBasePtr(), qOff);
          emitOpaqueCallVoid(rewriter, loc, "__riscv_vse8_v_i8m2",
                             mlir::ValueRange{qsOffPtr, vQ, vl}, opName, role);

          // bsums: the first 16-lane chunk BEFORE the inner loop, then advance by
          // vslidedown 16 lanes at a time. Each chunk: vget the low i8m1 register,
          // vwredsum over 16 lanes seeded 0, extract (int16_t), store bsums[idx].
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "bsums_first"));
          mlir::Value sixteen = sizeLit(16);
          mlir::Value firstIdx =
              rewriter.create<emitc::DivOp>(loc, sizeType, qOff, sixteen);
          mlir::Value chunk0 = vcall(i8m1Type, "__riscv_vget_v_i8m2_i8m1",
                                     mlir::ValueRange{vQ, zero});
          mlir::Value vSum0 =
              vcall(i16m1Type, "__riscv_vwredsum_vs_i8m1_i16m1",
                    mlir::ValueRange{chunk0, zeroSum, sixteen});
          mlir::Value sum0 = vcall(i16Type, "__riscv_vmv_x_s_i16m1_i16",
                                   mlir::ValueRange{vSum0});
          storeBsum(firstIdx, sum0);

          // vint8m2_t slid_q = v_q;  (the loop-carried slidedown register).
          auto slidVar = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i8m2Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(loc, slidVar, vQ);

          // for (size_t k = 16; k < vl; k += 16) {
          //   slid_q = vslidedown_vx_i8m2(slid_q, 16, vl);
          //   idx = (off + k) / 16;
          //   chunk = vget_v_i8m2_i8m1(slid_q, 0);
          //   bsums[idx] = vmv_x_s(vwredsum(chunk, v_zero_sum, 16));
          // }
          auto bsumsFor = rewriter.create<emitc::ForOp>(loc, sixteen, vl,
                                                        sixteen,
                                                        /*bodyBuilder=*/nullptr);
          mlir::Value kIv = bsumsFor.getInductionVar();
          {
            mlir::OpBuilder::InsertionGuard g3(rewriter);
            rewriter.setInsertionPointToStart(bsumsFor.getBody());
            mlir::Value slidCur =
                rewriter.create<emitc::LoadOp>(loc, i8m2Type, slidVar)
                    .getResult();
            mlir::Value slidNext =
                vcall(i8m2Type, "__riscv_vslidedown_vx_i8m2",
                      mlir::ValueRange{slidCur, sixteen, vl});
            rewriter.create<emitc::AssignOp>(loc, slidVar, slidNext);
            mlir::Value offK =
                rewriter.create<emitc::AddOp>(loc, sizeType, qOff, kIv);
            mlir::Value idx =
                rewriter.create<emitc::DivOp>(loc, sizeType, offK, sixteen);
            mlir::Value chunk = vcall(i8m1Type, "__riscv_vget_v_i8m2_i8m1",
                                      mlir::ValueRange{slidNext, zero});
            mlir::Value vSum =
                vcall(i16m1Type, "__riscv_vwredsum_vs_i8m1_i16m1",
                      mlir::ValueRange{chunk, zeroSum, sixteen});
            mlir::Value sumK = vcall(i16Type, "__riscv_vmv_x_s_i16m1_i16",
                                     mlir::ValueRange{vSum});
            storeBsum(idx, sumK);
          }
        }
        rewriter.create<emitc::YieldOp>(loc);
      }
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseRopeRotateStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::TypedElementwiseLoopBodyOp loopBody,
    tcrvrvv::ElementwiseRopeRotateCoreOp ropeCore, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The CONSTRUCTED rope rotate-model body, the per-pair-recurrence sibling of
    // the map paths (scale/silu) and the reduce paths (rms_norm/soft_max). The
    // outer loop op owns the rotate shape (reduce_map_model "rotate": a per-pair
    // scalar loop with a loop-carried f32 theta region arg + the yield that
    // carries it back); this re-emit sources the WHOLE rope ABI + the byte-exact
    // scalar cos/sin angle seam / position-dependent 2x2 rotation / f32 theta
    // recurrence from the region's rotate core brick (anti-bypass). BYTE-EXACT to
    // the retired monolith tcrv_rvv.ggml_rope_norm_f32 emit modulo ONLY the
    // source-op provenance token.

    // Anti-bypass (I7): the brick's pair_index MUST be the loop induction variable
    // (region arg 0) and its theta MUST be the loop-carried recurrence (region
    // arg 1); the verifier pins both, checked here fail-closed too.
    mlir::Block &block = loopBody.getBody().front();
    if (block.getNumArguments() < 2 ||
        ropeCore.getPairIndex() != block.getArgument(0) ||
        ropeCore.getTheta() != block.getArgument(1))
      return rewriter.notifyMatchFailure(
          ropeCore, "the rope rotate core brick's pair_index / theta must be the "
                    "loop induction variable / loop-carried recurrence (region "
                    "args 0 / 1)");

    mlir::Value input = valueMap.lookup(ropeCore.getInput());
    mlir::Value output = valueMap.lookup(ropeCore.getOutput());
    mlir::Value thetaBase = valueMap.lookup(ropeCore.getThetaBase());
    mlir::Value thetaScale = valueMap.lookup(ropeCore.getThetaScale());
    if (!input || !output || !thetaBase || !thetaScale)
      return rewriter.notifyMatchFailure(ropeCore, "rope ABI operand unmapped");

    llvm::StringRef opName = ropeCore.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = ropeCore.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type indexType = rewriter.getIndexType();

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float theta = theta_base;  -- the iterative f32 angle recurrence seed
    // (ggml's `float theta = theta_base;` ops.cpp:5711). emitc.for has no
    // iter_args, so the loop-carried theta is an emitc.variable lvalue +
    // emitc.assign, exactly as F3 (rms_norm) carries its scalar-double sum.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("theta", opName, role));
    auto thetaVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(loc, thetaVar, thetaBase);

    // size_t n_pairs = n_dims / 2;  (ggml steps i0 by 2 over [0, ne0)).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "pair_count"));
    mlir::Value nPairs =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(2));

    // for (size_t p = 0; p < n_pairs; p += 1) { ... }  -- the SCALAR per-pair
    // rotation loop. NOT vectorized: cos/sin are scalar libm (one call per pair),
    // so the faithful structure IS ggml's scalar loop.
    mlir::Value zero = sizeLit(0);
    mlir::Value one = sizeLit(1);
    auto pairFor = rewriter.create<emitc::ForOp>(loc, zero, nPairs, one,
                                                 /*bodyBuilder=*/nullptr);
    mlir::Value p = pairFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(pairFor.getBody());

      // float theta_p = theta;  (read the loop-carried recurrence value).
      mlir::Value thetaCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, thetaVar).getResult();

      // float cos_t = cosf(theta_p);  float sin_t = sinf(theta_p);  -- the SCALAR
      // libm angle cache (ggml's rope_yarn cosf/sinf, ops.cpp:5703-5704). Each is
      // ONE emitc.call_opaque (the sanctioned opaque seam) -- the byte-exactness
      // axis that depends on linking the SAME libm ggml links (NOT a raw string,
      // NOT a vectorized polynomial).
      mlir::Value cosT = emitOpaqueCall(rewriter, loc, floatType, "cosf",
                                        mlir::ValueRange{thetaCur}, opName, role);
      mlir::Value sinT = emitOpaqueCall(rewriter, loc, floatType, "sinf",
                                        mlir::ValueRange{thetaCur}, opName, role);

      // const float *xp = (const float *)(x + 2*p);  float *yp = (float *)(y+2*p)
      // -- the CONSECUTIVE pair (NORMAL: ic = i0, x0=x[2p], x1=x[2p+1]).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "pair_ptr"));
      mlir::Value pairOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, p, sizeLit(2));
      mlir::Value xpRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, pairOff);
      auto xp = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xpRaw)
              .getResult());
      mlir::Value ypRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, pairOff);
      auto yp = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, floatPtrType, ypRaw).getResult());

      // float x0 = xp[0];  float x1 = xp[1];  -- the consecutive pair loads.
      mlir::Value idx0 = rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
      mlir::Value idx1 = rewriter.create<emitc::LiteralOp>(loc, indexType, "1");
      // The subscript on a `const float *` yields an lvalue of `const float`; the
      // load must take that exact value type (a fixed `float` mismatches the
      // emitc.load verifier). The loaded scalar then participates as a plain f32.
      emitc::SubscriptOp x0Sub =
          rewriter.create<emitc::SubscriptOp>(loc, xp, idx0);
      auto xLValueType =
          llvm::cast<emitc::LValueType>(x0Sub.getResult().getType());
      mlir::Value x0 =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     x0Sub.getResult())
              .getResult();
      emitc::SubscriptOp x1Sub =
          rewriter.create<emitc::SubscriptOp>(loc, xp, idx1);
      mlir::Value x1 =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     x1Sub.getResult())
              .getResult();

      // The ROTATION. Each output's `a*b - c*d` / `a*b + c*d` is grouped into ONE
      // emitc.expression, so mlir-translate renders it as ONE C statement
      // TOKEN-IDENTICAL to ggml's source (ops.cpp:5808-5809 -- the rotation is a
      // single C expression there). Then clang makes the IDENTICAL contraction
      // decision under EVERY -ffp-contract mode (fuses under on/fast, two-rounding
      // under off) -- so the kernel is byte-exact vs ggml regardless of the build
      // flag, NOT only at off. Emitting the two products as separate statements
      // would block intra-statement fusion and diverge from ggml's fused form
      // under the default `on`. The subscript-loads and the cosf/sinf
      // call_opaque results stay OUTSIDE the expression (load/call_opaque lack the
      // CExpression trait). This mirrors the F3 rms_norm emitc.expression FMA fix.
      mlir::Value idx0Lit =
          rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
      mlir::Value idx1Lit =
          rewriter.create<emitc::LiteralOp>(loc, indexType, "1");
      // yp[0] = x0*cos_t - x1*sin_t;  (one expression -> one C statement)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "rotate_lo"));
      auto loExpr = rewriter.create<emitc::ExpressionOp>(loc, floatType,
                                                         /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&loExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value x0cos =
            rewriter.create<emitc::MulOp>(loc, floatType, x0, cosT);
        mlir::Value x1sin =
            rewriter.create<emitc::MulOp>(loc, floatType, x1, sinT);
        mlir::Value y0 =
            rewriter.create<emitc::SubOp>(loc, floatType, x0cos, x1sin);
        rewriter.create<emitc::YieldOp>(loc, y0);
      }
      // yp is a DISTINCT subscript chain from xp (the loads addressed the const
      // xp; the stores address the mutable yp -- ggml writes dst, reads src,
      // which may alias but the consecutive pair is fully READ before written).
      emitc::SubscriptOp y0Sub =
          rewriter.create<emitc::SubscriptOp>(loc, yp, idx0Lit);
      rewriter.create<emitc::AssignOp>(loc, y0Sub.getResult(),
                                       loExpr.getResult());

      // yp[1] = x0*sin_t + x1*cos_t;  (one expression -> one C statement)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "rotate_hi"));
      auto hiExpr = rewriter.create<emitc::ExpressionOp>(loc, floatType,
                                                         /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&hiExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value x0sin =
            rewriter.create<emitc::MulOp>(loc, floatType, x0, sinT);
        mlir::Value x1cos =
            rewriter.create<emitc::MulOp>(loc, floatType, x1, cosT);
        mlir::Value y1 =
            rewriter.create<emitc::AddOp>(loc, floatType, x0sin, x1cos);
        rewriter.create<emitc::YieldOp>(loc, y1);
      }
      emitc::SubscriptOp y1Sub =
          rewriter.create<emitc::SubscriptOp>(loc, yp, idx1Lit);
      rewriter.create<emitc::AssignOp>(loc, y1Sub.getResult(),
                                       hiExpr.getResult());

      // theta = theta * theta_scale;  -- the iterative f32 recurrence step
      // (ggml's `theta *= theta_scale;` ops.cpp:5719).
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("theta", opName, role));
      mlir::Value thetaNext =
          rewriter.create<emitc::MulOp>(loc, floatType, thetaCur, thetaScale);
      rewriter.create<emitc::AssignOp>(loc, thetaVar, thetaNext);
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlForwardElementwiseF32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The DISPATCH-WIRED forward-elementwise f32 support body (add/mul/cpy/gelu):
    // a hand-written monolith emit, NOT a constructed typed loop brick ([L-6]
    // wiring != construction). The recognizer guarantees the with_vl body is
    // EXACTLY one of the four ops; find it and route by op identity.
    mlir::Operation *fwd = nullptr;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlVecAddF32Op, tcrvrvv::GgmlVecMulF32Op,
                    tcrvrvv::GgmlVecCpyF32Op, tcrvrvv::GgmlGeluF32Op>(op))
        fwd = &op;
    }
    if (!fwd)
      return rewriter.notifyMatchFailure(
          scope, "forward-elementwise body missing the op");

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type indexType = rewriter.getIndexType();

    // The support ops pin the m8 strip anchor ggml's apply path uses. The bare
    // per-lane add/mul/cpy are byte-exact at any LMUL (no reduction), so m8 is a
    // fixed resource fact, not a knob (matching silu/quantize precedent).
    llvm::StringRef lmul = "m8";
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
    std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");

    // The vectorized m8 strip loop for the bare per-lane maps (add/mul/cpy): load
    // each input strip, combine (or pass through, for cpy), store the output
    // strip. `binaryCallee` is empty for cpy (a pure load->store copy).
    auto emitVectorStrip =
        [&](mlir::ValueRange inputs, mlir::Value output, llvm::StringRef opName,
            llvm::StringRef role,
            llvm::StringRef binaryCallee) -> mlir::LogicalResult {
      rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

      // size_t vlmax = __riscv_vsetvl_e32m8(n);
      mlir::Value vlmax = emitOpaqueCall(rewriter, loc, sizeType, setvlCallee,
                                         mlir::ValueRange{avlArg}, opName, role);
      // for (size_t i = 0; i < n; i += vlmax) { ... }
      mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
      auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                                 /*bodyBuilder=*/nullptr);
      mlir::Value iv = forOp.getInductionVar();
      {
        mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());

        // size_t vl = __riscv_vsetvl_e32m8(n - i);
        mlir::Value bodyVL = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, setvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining =
                  b.create<emitc::SubOp>(l, sizeType, avlArg, iv);
              return {remaining};
            });

        // vfloat32m8_t v_k = __riscv_vle32_v_f32m8((const float *)(in_k + i), vl);
        llvm::SmallVector<mlir::Value> loaded;
        for (mlir::Value in : inputs) {
          mlir::Value pRaw =
              rewriter.create<emitc::AddOp>(loc, in.getType(), in, iv);
          mlir::Value p =
              rewriter.create<emitc::CastOp>(loc, constFloatPtrType, pRaw)
                  .getResult();
          loaded.push_back(emitOpaqueCall(rewriter, loc, f32VecType, loadCallee,
                                          mlir::ValueRange{p, bodyVL}, opName,
                                          role));
        }

        // v_out = __riscv_vfadd_vv_f32m8 | __riscv_vfmul_vv_f32m8 (or the loaded
        // strip itself for the copy).
        mlir::Value result;
        if (binaryCallee.empty()) {
          result = loaded.front();
        } else {
          result = emitOpaqueCall(
              rewriter, loc, f32VecType, binaryCallee,
              mlir::ValueRange{loaded[0], loaded[1], bodyVL}, opName, role);
        }

        // __riscv_vse32_v_f32m8((float *)(out + i), v_out, vl);
        mlir::Value oRaw =
            rewriter.create<emitc::AddOp>(loc, output.getType(), output, iv);
        mlir::Value o =
            rewriter.create<emitc::CastOp>(loc, floatPtrType, oRaw).getResult();
        emitOpaqueCallVoid(rewriter, loc, storeCallee,
                           mlir::ValueRange{o, result, bodyVL}, opName, role);
      }
      return mlir::success();
    };

    if (auto addOp = llvm::dyn_cast<tcrvrvv::GgmlVecAddF32Op>(fwd)) {
      mlir::Value lhs = valueMap.lookup(addOp.getLhs());
      mlir::Value rhs = valueMap.lookup(addOp.getRhs());
      mlir::Value output = valueMap.lookup(addOp.getOutput());
      if (!lhs || !rhs || !output)
        return rewriter.notifyMatchFailure(addOp, "vec_add ABI operand unmapped");
      return emitVectorStrip({lhs, rhs}, output,
                             addOp.getTCRVEmitCLowerableSourceOpName(),
                             addOp.getTCRVEmitCLowerableSourceRole(),
                             riscvIntrinsicName("vfadd", 32, lmul, "f32"));
    }
    if (auto mulOp = llvm::dyn_cast<tcrvrvv::GgmlVecMulF32Op>(fwd)) {
      mlir::Value lhs = valueMap.lookup(mulOp.getLhs());
      mlir::Value rhs = valueMap.lookup(mulOp.getRhs());
      mlir::Value output = valueMap.lookup(mulOp.getOutput());
      if (!lhs || !rhs || !output)
        return rewriter.notifyMatchFailure(mulOp, "vec_mul ABI operand unmapped");
      return emitVectorStrip({lhs, rhs}, output,
                             mulOp.getTCRVEmitCLowerableSourceOpName(),
                             mulOp.getTCRVEmitCLowerableSourceRole(),
                             riscvIntrinsicName("vfmul", 32, lmul, "f32"));
    }
    if (auto cpyOp = llvm::dyn_cast<tcrvrvv::GgmlVecCpyF32Op>(fwd)) {
      mlir::Value input = valueMap.lookup(cpyOp.getInput());
      mlir::Value output = valueMap.lookup(cpyOp.getOutput());
      if (!input || !output)
        return rewriter.notifyMatchFailure(cpyOp, "vec_cpy ABI operand unmapped");
      return emitVectorStrip({input}, output,
                             cpyOp.getTCRVEmitCLowerableSourceOpName(),
                             cpyOp.getTCRVEmitCLowerableSourceRole(),
                             /*binaryCallee=*/"");
    }

    // gelu: the SCALAR per-element tanh gelu loop. tanhf is the sanctioned
    // scalar-libm opaque seam (the sibling of rope's cosf/sinf, rms_norm's sqrtf),
    // so the faithful thin body is ggml's reference formula, one call per element.
    auto geluOp = llvm::cast<tcrvrvv::GgmlGeluF32Op>(fwd);
    mlir::Value input = valueMap.lookup(geluOp.getInput());
    mlir::Value output = valueMap.lookup(geluOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(geluOp, "gelu ABI operand unmapped");
    llvm::StringRef opName = geluOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = geluOp.getTCRVEmitCLowerableSourceRole();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // for (size_t i = 0; i < n; i += 1) { y[i] = gelu(x[i]); }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value one = rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, one,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value iv = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // const float *xp = (const float *)(x + i);  float x_i = xp[0];
      mlir::Value xpRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, iv);
      auto xp = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xpRaw)
              .getResult());
      mlir::Value idx0 = rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
      emitc::SubscriptOp xSub =
          rewriter.create<emitc::SubscriptOp>(loc, xp, idx0);
      auto xLValueType =
          llvm::cast<emitc::LValueType>(xSub.getResult().getType());
      mlir::Value xv =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     xSub.getResult())
              .getResult();

      // The ggml reference tanh gelu (ggml_gelu_f32):
      //   0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))
      mlir::Value oneF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
      mlir::Value halfF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.5f");
      mlir::Value coefA =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.044715f");
      mlir::Value sqrt2pi = rewriter.create<emitc::LiteralOp>(
          loc, floatType, "0.79788456080286535587989211986876f");
      mlir::Value x2 = rewriter.create<emitc::MulOp>(loc, floatType, xv, xv);
      mlir::Value coefX2 =
          rewriter.create<emitc::MulOp>(loc, floatType, coefA, x2);
      mlir::Value innerA =
          rewriter.create<emitc::AddOp>(loc, floatType, oneF, coefX2);
      mlir::Value sqrtX =
          rewriter.create<emitc::MulOp>(loc, floatType, sqrt2pi, xv);
      mlir::Value inner =
          rewriter.create<emitc::MulOp>(loc, floatType, sqrtX, innerA);
      mlir::Value tanhV = emitOpaqueCall(rewriter, loc, floatType, "tanhf",
                                         mlir::ValueRange{inner}, opName, role);
      mlir::Value onePlusTanh =
          rewriter.create<emitc::AddOp>(loc, floatType, oneF, tanhV);
      mlir::Value halfX =
          rewriter.create<emitc::MulOp>(loc, floatType, halfF, xv);
      mlir::Value gv =
          rewriter.create<emitc::MulOp>(loc, floatType, halfX, onePlusTanh);

      // float *yp = (float *)(y + i);  yp[0] = gelu(x_i);
      mlir::Value ypRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, iv);
      auto yp = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, floatPtrType, ypRaw).getResult());
      mlir::Value idx0y = rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
      emitc::SubscriptOp ySub =
          rewriter.create<emitc::SubscriptOp>(loc, yp, idx0y);
      rewriter.create<emitc::AssignOp>(loc, ySub.getResult(), gv);
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlDequantizeRow(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The DISPATCH-WIRED dequantize_row family body: the single
    // tcrv_rvv.dequantize_row op's bounded `format` routes to a hand-written AoS
    // block-decode ([L-6] wiring != construction). The decode is byte-exact to
    // ggml's reference dequantize_row_<format> (a scalar block loop); the per-strip
    // f32 stores need no reduction, so LMUL/strip-count are correctness-free.
    tcrvrvv::GgmlDequantizeRowOp deqOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto d = llvm::dyn_cast<tcrvrvv::GgmlDequantizeRowOp>(op))
        deqOp = d;
    }
    if (!deqOp)
      return rewriter.notifyMatchFailure(scope, "dequant body missing the op");

    mlir::Value input = valueMap.lookup(deqOp.getInput());
    mlir::Value output = valueMap.lookup(deqOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(deqOp, "dequant ABI operand unmapped");

    llvm::StringRef format = deqOp.getFormat();
    llvm::StringRef opName = deqOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = deqOp.getTCRVEmitCLowerableSourceRole();

    // The EXTENDED formats (K-quant super-blocks / FP4 codebooks / ternary /
    // iq4_nl) each route to their own hand-written scalar super-block decode
    // (byte-exact to ggml's reference dequantize_row_<format>, reusing that
    // format's already-constructed vec_dot block-decode facts). The extended
    // dispatch returns a plain match failure (no ops emitted) for a legacy
    // format, so the legacy nibble/int8 chain below still owns q4_0..q8_0.
    if (mlir::succeeded(emitGgmlDequantizeRowExtended(
            rewriter, loc, deqOp, input, output, avlArg, sizeType, opName, role)))
      return mlir::success();

    // Per-format AoS block layout facts (ggml-common.h + QK*_0/1 = 32). offsets are
    // byte offsets into the AoS block; qsElem is the qs element ctype (int8 for the
    // bare-scale q8_0, uint8 nibble carrier otherwise). `sub` = the pre-scale bias
    // subtracted from the nibble (8 for q4_0, 16 for q5_0), 0 when a min is added
    // instead. hasMin / hasQh gate the q4_1/q5_1 min and the q5_0/q5_1 5th bit.
    int64_t stride = 0, dOff = 0, mOff = 0, qhOff = 0, qsOff = 0, sub = 0;
    bool hasMin = false, hasQh = false, bareInt8 = false;
    if (format == "q4_0") {
      stride = 18; dOff = 0; qsOff = 2; sub = 8;
    } else if (format == "q4_1") {
      stride = 20; dOff = 0; mOff = 2; qsOff = 4; hasMin = true;
    } else if (format == "q5_0") {
      stride = 22; dOff = 0; qhOff = 2; qsOff = 6; sub = 16; hasQh = true;
    } else if (format == "q5_1") {
      stride = 24; dOff = 0; mOff = 2; qhOff = 4; qsOff = 8;
      hasMin = true; hasQh = true;
    } else if (format == "q8_0") {
      stride = 34; dOff = 0; qsOff = 2; bareInt8 = true;
    } else {
      return rewriter.notifyMatchFailure(deqOp,
                                         "unwired dequantize_row format");
    }

    // q8_0 is the FRONT-DOOR CONSTRUCTED family-head: in production the emit driver
    // routes the abstract q8_0 op to constructOrEmitGgmlDequantizeRow (which builds
    // the typed_dequantize_row_loop_body region and lowers it via the SHARED body
    // emitter). This defensive early-return keeps the SAME shared body as the single
    // q8_0 emit source, so the dispatch-wired and constructed paths never diverge.
    if (bareInt8)
      return emitDequantizeRowQ8_0BodyShared(rewriter, loc, input, output, avlArg,
                                             sizeType, opName, role);

    // The 4-bit nibble formats (q4_0/q4_1/q5_0/q5_1) route to the SHARED nibble
    // decode body -- the SAME emitter the front-door CONSTRUCTED per-format leaves
    // invoke, so the dispatch-wired fallback and the constructed lowering emit
    // byte-identical C (modulo only the source-op provenance token). Byte-exact by
    // construction.
    return emitDequantizeRowNibbleBodyShared(
        rewriter, loc, input, output, avlArg, sizeType, opName, role, stride, dOff,
        mOff, qhOff, qsOff, sub, hasMin, hasQh);
  }

// The SHARED 4-bit nibble dequantize_row block-decode body (q4_0/q4_1/q5_0/q5_1):
// the AoS nb=k/32 block loop, the fp16 d (+ optional fp16 min m) seam, the optional
// byte-assembled uint32 qh 5th-bit plane, then the per-j nibble unpack with the
// optional 5th-bit merge, the pre-scale bias / min add, and the f32 scale. Extracted
// VERBATIM from the nibble tail of emitGgmlDequantizeRow so the dispatch-wired
// monolith fallback and the CONSTRUCTED typed lowering (via the per-format leaves)
// emit byte-identical C. Byte-exact to ggml's reference dequantize_row_<format>.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowNibbleBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    int64_t stride, int64_t dOff, int64_t mOff, int64_t qhOff, int64_t qsOff,
    int64_t sub, bool hasMin, bool hasQh) const {
    // block_q4_0/q4_1/q5_0/q5_1 AoS facts: all carry qk=32 lanes per block.
    const int64_t qk = 32;
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type indexType = rewriter.getIndexType();
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto intLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, intType, std::to_string(v));
    };
    auto uintLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, uintType,
                                               std::to_string(v) + "u");
    };
    auto idxLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, indexType,
                                               std::to_string(v));
    };
    // Load *elemPtr (an ALREADY pointer-advanced typed pointer, so the address is
    // `base + j` -- the proven gelu idiom of pointer arithmetic + subscript[0],
    // avoiding an induction-var subscript) and widen to `int` (signed int8
    // sign-extends, uint8 zero-extends).
    auto loadElemAsInt = [&](mlir::Value elemPtr,
                             mlir::Type elemType) -> mlir::Value {
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(elemPtr),
                  idxLit(0))
              .getResult();
      mlir::Value v =
          rewriter.create<emitc::LoadOp>(loc, elemType, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
    };
    auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr),
                  idxLit(i))
              .getResult();
      mlir::Value v =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, uintType, v).getResult();
    };
    // *elemPtr = value  (store one f32 through an ALREADY pointer-advanced float*).
    auto storeF32 = [&](mlir::Value elemPtr, mlir::Value value) {
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(elemPtr),
                  idxLit(0))
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, elem, value);
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = k / 32;  (ggml's `const int nb = k / qk`; k % qk == 0 is a ggml
    // contract, no tail).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the AoS block loop.
    mlir::Value zero = sizeLit(0);
    mlir::Value one = sizeLit(1);
    auto blockFor = rewriter.create<emitc::ForOp>(loc, zero, nb, one,
                                                  /*bodyBuilder=*/nullptr);
    mlir::Value ib = blockFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(blockFor.getBody());

      // const uint8_t *xb = x + ib*stride;  (the AoS block byte cursor).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "x_block"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      mlir::Value xb =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);

      // float *yb = y + ib*32;  (the f32 output row block).
      mlir::Value yOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk));
      mlir::Value ybRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, yOff);
      mlir::Value yb =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

      // float d = (float)*(const _Float16 *)(xb + dOff);  (the fp16 block scale;
      // the board is __riscv_zfhmin so this is fcvt.s.h).
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
      mlir::Value dAddr = xb;
      if (dOff != 0)
        dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb,
                                              sizeLit(dOff));
      mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                     mlir::ValueRange{dAddr}, opName, role,
                                     llvm::StringRef("fcvt.s.h"));

      // float m = (float)*(const _Float16 *)(xb + mOff);  (q4_1/q5_1 min).
      mlir::Value m;
      if (hasMin) {
        mlir::Value mAddr =
            rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(mOff));
        m = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                           mlir::ValueRange{mAddr}, opName, role,
                           llvm::StringRef("fcvt.s.h"));
      }

      // uint32_t qh = qh[0] | qh[1]<<8 | qh[2]<<16 | qh[3]<<24;  (q5_0/q5_1 5th-bit
      // plane; the byte-assembled little-endian load matches ggml's memcpy(&qh)).
      mlir::Value qh;
      if (hasQh) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh"));
        mlir::Value qhBaseRaw =
            rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qhOff));
        mlir::Value qhBase =
            rewriter.create<emitc::CastOp>(loc, u8PtrType, qhBaseRaw)
                .getResult();
        mlir::Value b0 = loadByteAsUint(qhBase, 0);
        mlir::Value b1 = loadByteAsUint(qhBase, 1);
        mlir::Value b2 = loadByteAsUint(qhBase, 2);
        mlir::Value b3 = loadByteAsUint(qhBase, 3);
        mlir::Value s1 = rewriter.create<emitc::BitwiseLeftShiftOp>(
            loc, uintType, b1, uintLit(8));
        mlir::Value s2 = rewriter.create<emitc::BitwiseLeftShiftOp>(
            loc, uintType, b2, uintLit(16));
        mlir::Value s3 = rewriter.create<emitc::BitwiseLeftShiftOp>(
            loc, uintType, b3, uintLit(24));
        qh = rewriter.create<emitc::BitwiseOrOp>(loc, uintType, b0, s1)
                 .getResult();
        qh = rewriter.create<emitc::BitwiseOrOp>(loc, uintType, qh, s2)
                 .getResult();
        qh = rewriter.create<emitc::BitwiseOrOp>(loc, uintType, qh, s3)
                 .getResult();
      }

      // const uint8_t *qs = (const uint8_t *)(xb + qsOff);  (the packed nibbles).
      mlir::Value qsBaseRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff));
      mlir::Value qsBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBaseRaw).getResult();

      // The 4-bit nibble formats (q4_0/q4_1/q5_0/q5_1): for (j = 0; j < 16; ++j)
      // decode the low nibble -> y[j] and the high nibble -> y[j+16].
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "nibble_decode"));
      auto nFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), sizeLit(qk / 2),
                                                sizeLit(1),
                                                /*bodyBuilder=*/nullptr);
      mlir::Value j = nFor.getInductionVar();
      {
        mlir::OpBuilder::InsertionGuard nGuard(rewriter);
        rewriter.setInsertionPointToStart(nFor.getBody());

        // int qi = qs[j];  int nlo = qi & 0x0F;  int nhi = qi >> 4;
        mlir::Value qp =
            rewriter.create<emitc::AddOp>(loc, u8PtrType, qsBase, j);
        mlir::Value qi = loadElemAsInt(qp, constU8Type);
        mlir::Value nlo =
            rewriter.create<emitc::BitwiseAndOp>(loc, intType, qi, intLit(15))
                .getResult();
        mlir::Value nhi =
            rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, qi,
                                                        intLit(4))
                .getResult();

        // The 5th-bit merge (q5_0/q5_1): xh0 = ((qh >> j) << 4) & 0x10;
        // xh1 = ((qh >> (j+12))) & 0x10;  then OR into the nibbles (int domain).
        if (hasQh) {
          // (uint32_t)j is the little-endian bit index into qh (ggml's j).
          mlir::Value jInt =
              rewriter.create<emitc::CastOp>(loc, uintType, j).getResult();
          mlir::Value jInt12 = rewriter.create<emitc::AddOp>(loc, uintType, jInt,
                                                             uintLit(12));
          mlir::Value r0 = rewriter.create<emitc::BitwiseRightShiftOp>(
              loc, uintType, qh, jInt).getResult();
          mlir::Value r0s = rewriter.create<emitc::BitwiseLeftShiftOp>(
              loc, uintType, r0, uintLit(4)).getResult();
          mlir::Value xh0u = rewriter.create<emitc::BitwiseAndOp>(
              loc, uintType, r0s, uintLit(16)).getResult();
          mlir::Value r1 = rewriter.create<emitc::BitwiseRightShiftOp>(
              loc, uintType, qh, jInt12).getResult();
          mlir::Value xh1u = rewriter.create<emitc::BitwiseAndOp>(
              loc, uintType, r1, uintLit(16)).getResult();
          mlir::Value xh0 =
              rewriter.create<emitc::CastOp>(loc, intType, xh0u).getResult();
          mlir::Value xh1 =
              rewriter.create<emitc::CastOp>(loc, intType, xh1u).getResult();
          nlo = rewriter.create<emitc::BitwiseOrOp>(loc, intType, nlo, xh0)
                    .getResult();
          nhi = rewriter.create<emitc::BitwiseOrOp>(loc, intType, nhi, xh1)
                    .getResult();
        }

        // The pre-scale bias: q4_0 -8, q5_0 -16 (subtracted); q4_1/q5_1 add m.
        if (sub != 0) {
          nlo = rewriter.create<emitc::SubOp>(loc, intType, nlo, intLit(sub))
                    .getResult();
          nhi = rewriter.create<emitc::SubOp>(loc, intType, nhi, intLit(sub))
                    .getResult();
        }

        mlir::Value nloF =
            rewriter.create<emitc::CastOp>(loc, floatType, nlo).getResult();
        mlir::Value nhiF =
            rewriter.create<emitc::CastOp>(loc, floatType, nhi).getResult();
        mlir::Value y0 = rewriter.create<emitc::MulOp>(loc, floatType, nloF, d);
        mlir::Value y1 = rewriter.create<emitc::MulOp>(loc, floatType, nhiF, d);
        if (hasMin) {
          y0 = rewriter.create<emitc::AddOp>(loc, floatType, y0, m).getResult();
          y1 = rewriter.create<emitc::AddOp>(loc, floatType, y1, m).getResult();
        }

        // float *yp0 = yb + j;  float *yp1 = yb + j + 16;  y[j]=y0; y[j+16]=y1;
        mlir::Value yp0 =
            rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, j);
        mlir::Value yp1 =
            rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, j);
        yp1 = rewriter.create<emitc::AddOp>(loc, floatPtrType, yp1,
                                            sizeLit(qk / 2));
        storeF32(yp0, y0);
        storeF32(yp1, y1);
      }
    }

    return mlir::success();
  }

// The SHARED q8_0 dequantize_row block-decode body: the AoS `nb = k/32` block loop,
// the fp16 block scale via the `(float)*(const _Float16 *)` seam, and the bare
// signed-int8 scale `y[j] = qs[j] * d` over all 32 block lanes (the load
// sign-extends). Extracted VERBATIM from the q8_0 branch of emitGgmlDequantizeRow so
// the DISPATCH-WIRED monolith fallback and the CONSTRUCTED typed lowering emit
// byte-identical C (modulo only the source-op provenance token threaded through
// opName/role). Byte-exact to ggml's reference dequantize_row_q8_0 (a scalar block
// loop; no reduction).
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ8_0BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  // block_q8_0 AoS facts: fp16 d @0, 32 signed int8 quants @2, stride 34.
  const int64_t qk = 32, stride = 34, qsOff = 2;

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();
  mlir::Type outputPtrType = output.getType();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
  mlir::Type i8PtrType = emitc::PointerType::get(constI8Type);
  mlir::Type floatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
  };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType,
                                             std::to_string(v));
  };
  auto loadElemAsInt = [&](mlir::Value elemPtr,
                           mlir::Type elemType) -> mlir::Value {
    mlir::Value elem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(elemPtr),
                idxLit(0))
            .getResult();
    mlir::Value v =
        rewriter.create<emitc::LoadOp>(loc, elemType, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };
  auto storeF32 = [&](mlir::Value elemPtr, mlir::Value value) {
    mlir::Value elem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(elemPtr),
                idxLit(0))
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, elem, value);
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // size_t nb = k / 32;  (ggml's `const int nb = k / qk`; k % qk == 0, no tail).
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  mlir::Value zero = sizeLit(0);
  mlir::Value one = sizeLit(1);
  auto blockFor = rewriter.create<emitc::ForOp>(loc, zero, nb, one,
                                                /*bodyBuilder=*/nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    // const uint8_t *xb = x + ib*34;  float *yb = y + ib*32;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "x_block"));
    mlir::Value xOff =
        rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
    mlir::Value xb =
        rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);
    mlir::Value yOff =
        rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk));
    mlir::Value ybRaw =
        rewriter.create<emitc::AddOp>(loc, outputPtrType, output, yOff);
    mlir::Value yb =
        rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    // float d = (float)*(const _Float16 *)xb;  (the fp16 block scale; fcvt.s.h).
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{xb}, opName, role,
                                   llvm::StringRef("fcvt.s.h"));

    // const int8_t *qs = (const int8_t *)(xb + 2);
    mlir::Value qsBaseRaw =
        rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff));
    mlir::Value qsBase =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, qsBaseRaw).getResult();

    // for (j = 0; j < 32; ++j) y[j] = qs[j] * d;  (bare signed-int8 scale).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "q8_scale"));
    auto qFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), sizeLit(qk),
                                              sizeLit(1),
                                              /*bodyBuilder=*/nullptr);
    mlir::Value j = qFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard qGuard(rewriter);
      rewriter.setInsertionPointToStart(qFor.getBody());
      mlir::Value qp =
          rewriter.create<emitc::AddOp>(loc, i8PtrType, qsBase, j);
      mlir::Value qi = loadElemAsInt(qp, constI8Type);
      mlir::Value qf =
          rewriter.create<emitc::CastOp>(loc, floatType, qi).getResult();
      mlir::Value yv = rewriter.create<emitc::MulOp>(loc, floatType, qf, d);
      mlir::Value yp = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, j);
      storeF32(yp, yv);
    }
  }

  return mlir::success();
}

// The per-format CONSTRUCTED dequantize_row decode leaves for the flat nibble family.
// Each hard-codes its ggml block_qX AoS layout facts (the byte-exact ABI shape
// constants, NOT tunable knobs) and calls the SHARED nibble body -- the SAME emitter
// the dispatch-wired monolith fallback invokes -- so the constructed lowering is
// byte-exact to the monolith by construction (modulo only the source-op provenance
// token). Streaming siblings of emitDequantizeRowQ8_0BodyShared (no accumulator).

// block_q4_0: fp16 d @0, 16 packed nibble bytes @2, stride 18; nibble bias -8.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ4_0BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleBodyShared(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/18, /*dOff=*/0, /*mOff=*/0, /*qhOff=*/0, /*qsOff=*/2,
      /*sub=*/8, /*hasMin=*/false, /*hasQh=*/false);
}

// block_q4_1: fp16 d @0, fp16 m @2, 16 packed nibble bytes @4, stride 20; min add.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ4_1BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleBodyShared(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/20, /*dOff=*/0, /*mOff=*/2, /*qhOff=*/0, /*qsOff=*/4,
      /*sub=*/0, /*hasMin=*/true, /*hasQh=*/false);
}

// block_q5_0: fp16 d @0, 4-byte qh 5th-bit plane @2, 16 nibble bytes @6, stride 22;
// nibble+5th-bit bias -16.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ5_0BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleBodyShared(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/22, /*dOff=*/0, /*mOff=*/0, /*qhOff=*/2, /*qsOff=*/6,
      /*sub=*/16, /*hasMin=*/false, /*hasQh=*/true);
}

// block_q5_1: fp16 d @0, fp16 m @2, 4-byte qh 5th-bit plane @4, 16 nibble bytes @8,
// stride 24; nibble+5th-bit with min add.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ5_1BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleBodyShared(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/24, /*dOff=*/0, /*mOff=*/2, /*qhOff=*/4, /*qsOff=*/8,
      /*sub=*/0, /*hasMin=*/true, /*hasQh=*/true);
}

// Lower the CONSTRUCTED streaming dequantize_row region: walk the
// tcrv_rvv.typed_dequantize_row_loop_body, extract its per-block DECODE brick
// (tcrv_rvv.dequantize_row_decode_core) + the VOID yield, enforce the anti-bypass
// invariant (the brick's block_index MUST be the region induction variable / region
// arg 0, so the ABI bases are sourced from the BRICK not inferred), and re-emit the
// whole nb block loop + per-block decode via the SHARED body emitter -- byte-exact to
// the dispatch-wired q8_0 monolith. The whole loop is emitter-inlined by the brick
// lowering (the streaming analog of q1_0/nvfp4's flat single-core-brick emit).
mlir::LogicalResult VariantToEmitCFunc::emitTypedDequantizeRowLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  tcrvrvv::TypedDequantizeRowLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<tcrvrvv::TypedDequantizeRowLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed dequantize_row loop body missing the op");

  // Bounded decode_model surface gate (I7): the constructed streaming family is
  // {q8_0 (family-head), q4_0/q4_1/q5_0/q5_1 (flat nibble leaves)}. The verifier
  // already gates decode_model; this fails the emit closed if a not-yet-lowered
  // decode leaf slips a valid-verify region here.
  llvm::StringRef decodeModel = loopBody.getDecodeModel();
  if (decodeModel != "q8_0" && decodeModel != "q4_0" && decodeModel != "q4_1" &&
      decodeModel != "q5_0" && decodeModel != "q5_1")
    return rewriter.notifyMatchFailure(
        loopBody, "typed dequantize_row loop body only lowers the constructed "
                  "streaming decode_models q8_0/q4_0/q4_1/q5_0/q5_1");

  tcrvrvv::DequantizeRowDecodeCoreOp coreOp;
  tcrvrvv::TypedDequantizeRowLoopYieldOp yieldOp;
  loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
    if (auto o = llvm::dyn_cast<tcrvrvv::DequantizeRowDecodeCoreOp>(bodyOp))
      coreOp = o;
    else if (auto o =
                 llvm::dyn_cast<tcrvrvv::TypedDequantizeRowLoopYieldOp>(bodyOp))
      yieldOp = o;
  });
  mlir::Block &coreBlock = loopBody.getBody().front();
  if (!coreOp || !yieldOp)
    return rewriter.notifyMatchFailure(
        loopBody, "typed dequantize_row body requires the "
                  "dequantize_row_decode_core brick + the void loop yield");
  if (coreBlock.getNumArguments() != 1)
    return rewriter.notifyMatchFailure(
        loopBody, "typed dequantize_row body region must carry exactly the "
                  "block_index induction variable");
  mlir::Value blockIndex = coreBlock.getArgument(0);
  if (coreOp.getBlockIndex() != blockIndex)
    return rewriter.notifyMatchFailure(
        loopBody, "the decode-core brick's block_index must be the loop "
                  "induction variable (region arg 0) so the emit addresses "
                  "base + ib*stride, not block-0");

  // Anti-bypass (I7): the ABI bases are sourced from the BRICK's operands.
  mlir::Value weightBase = valueMap.lookup(coreOp.getWeightBase());
  mlir::Value output = valueMap.lookup(coreOp.getOutput());
  if (!weightBase || !output)
    return rewriter.notifyMatchFailure(loopBody,
                                       "typed dequantize_row ABI operand unmapped");

  llvm::StringRef opName = loopBody.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getTCRVEmitCLowerableSourceRole();
  // Dispatch on decode_model to the per-format leaf: each re-emits the whole nb
  // block loop + per-block decode via the SHARED body emitter, byte-exact to the
  // dispatch-wired monolith.
  if (decodeModel == "q4_0")
    return emitDequantizeRowQ4_0BodyShared(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  if (decodeModel == "q4_1")
    return emitDequantizeRowQ4_1BodyShared(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  if (decodeModel == "q5_0")
    return emitDequantizeRowQ5_0BodyShared(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  if (decodeModel == "q5_1")
    return emitDequantizeRowQ5_1BodyShared(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  return emitDequantizeRowQ8_0BodyShared(rewriter, loc, weightBase, output,
                                         avlArg, sizeType, opName, role);
}

// The dequant FRONT DOOR (the flat streaming family {q8_0 family-head + the
// q4_0/q4_1/q5_0/q5_1 nibble leaves}): CONSTRUCT the typed
// tcrv_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core;
// typed_dequantize_row_loop_yield } in place of the abstract tcrv_rvv.dequantize_row,
// then LOWER it via emitTypedDequantizeRowLoopBody. The construction is a genuine IR
// rewrite (the emission is DRIVEN by the typed region op-identity + decode_model, not
// the abstract format string), so these flat formats are CONSTRUCTED ([L-6]/[L-8]),
// not dispatch-wired. The other (K-quant / FP4 / ternary / codebook / IQ) formats
// fall through to the dispatch-wired monolith.
mlir::LogicalResult VariantToEmitCFunc::constructOrEmitGgmlDequantizeRow(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  tcrvrvv::GgmlDequantizeRowOp deqOp;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto d = llvm::dyn_cast<tcrvrvv::GgmlDequantizeRowOp>(op))
      deqOp = d;
  }
  if (!deqOp)
    return rewriter.notifyMatchFailure(scope, "dequant body missing the op");

  // The flat streaming CONSTRUCTED family + its per-format AoS block-layout facts
  // (the ggml ABI shape constants, NOT tunable knobs): the block stride, the fp16
  // scale byte offset (scale_byte_offset), and the packed-quant byte offset
  // (quant_byte_offset). The min/qh offsets (q4_1/q5_1/q5_0) are baked into the
  // per-format decode leaf, so the brick carries only the two byte offsets the
  // shared q8_0-shaped attr surface names.
  llvm::StringRef format = deqOp.getFormat();
  int64_t qk = 32, stride = 0, dOff = 0, qsOff = 0;
  if (format == "q8_0") {
    stride = 34; qsOff = 2;
  } else if (format == "q4_0") {
    stride = 18; qsOff = 2;
  } else if (format == "q4_1") {
    stride = 20; qsOff = 4;
  } else if (format == "q5_0") {
    stride = 22; qsOff = 6;
  } else if (format == "q5_1") {
    stride = 24; qsOff = 8;
  } else {
    // The non-flat formats stay DISPATCH-WIRED (hand-written monolith).
    return emitGgmlDequantizeRow(rewriter, loc, scope, avlArg, sizeType,
                                 valueMap);
  }

  mlir::Value input = deqOp.getInput();
  mlir::Value output = deqOp.getOutput();
  mlir::Value n = deqOp.getElementCount();
  mlir::Type indexType = rewriter.getIndexType();

  {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(deqOp);

    mlir::OperationState loopState(
        loc, tcrvrvv::TypedDequantizeRowLoopBodyOp::getOperationName());
    loopState.addOperands({input, output, n});
    loopState.addAttribute(
        "kind", rewriter.getStringAttr("typed_dequantize_row_loop_body"));
    loopState.addAttribute("qk", rewriter.getI64IntegerAttr(qk));
    loopState.addAttribute("weight_block_stride",
                           rewriter.getI64IntegerAttr(stride));
    loopState.addAttribute("decode_model", rewriter.getStringAttr(format));
    loopState.addRegion();
    auto loopBody = llvm::cast<tcrvrvv::TypedDequantizeRowLoopBodyOp>(
        rewriter.create(loopState));

    mlir::Block *block = rewriter.createBlock(
        &loopBody.getBody(), loopBody.getBody().end(), {indexType}, {loc});
    mlir::Value blockIndex = block->getArgument(0);
    rewriter.setInsertionPointToStart(block);

    mlir::OperationState coreState(
        loc, tcrvrvv::DequantizeRowDecodeCoreOp::getOperationName());
    coreState.addOperands({input, output, blockIndex});
    coreState.addAttribute("decode_model", rewriter.getStringAttr(format));
    coreState.addAttribute("qk", rewriter.getI64IntegerAttr(qk));
    coreState.addAttribute("weight_block_stride",
                           rewriter.getI64IntegerAttr(stride));
    coreState.addAttribute("scale_byte_offset",
                           rewriter.getI64IntegerAttr(dOff));
    coreState.addAttribute("quant_byte_offset",
                           rewriter.getI64IntegerAttr(qsOff));
    rewriter.create(coreState);
    rewriter.create<tcrvrvv::TypedDequantizeRowLoopYieldOp>(loc);
  }
  rewriter.eraseOp(deqOp);

  return emitTypedDequantizeRowLoopBody(rewriter, loc, scope, avlArg, sizeType,
                                        valueMap);
}

// The EXTENDED dequantize_row decode: K-quant super-blocks, FP4 codebooks,
// ternary, and iq4_nl. Each body is a hand-written scalar AoS super-block loop
// reproducing ggml's reference dequantize_row_<format> byte-exactly (quants.c),
// reusing the SAME per-format block-decode facts already constructed for that
// format's block-dot vec_dot (the fp16 seam, the E8M0/UE4M3 scale, the
// get_scale_min_k4 6-bit unpack, the q3_K aux 6-bit scale shuffle, the base-3
// tq1_0 unpack, the 16-entry codebook gather). [L-6] wiring != construction.
mlir::LogicalResult VariantToEmitCFunc::emitGgmlDequantizeRowExtended(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::GgmlDequantizeRowOp deqOp, mlir::Value input, mlir::Value output,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role) const {
  llvm::StringRef format = deqOp.getFormat();

  enum Fmt {
    Q2K, Q3K, Q4K, Q5K, Q6K, MXFP4, NVFP4, TQ1, TQ2, IQ4NL,
    // The 8 IQ grid-table formats (each reuses its block-dot vec_dot grid decl).
    IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S, IQ1S, IQ1M, IQ4XS, NONE
  };
  Fmt fmt = llvm::StringSwitch<Fmt>(format)
                .Case("q2_K", Q2K)
                .Case("q3_K", Q3K)
                .Case("q4_K", Q4K)
                .Case("q5_K", Q5K)
                .Case("q6_K", Q6K)
                .Case("mxfp4", MXFP4)
                .Case("nvfp4", NVFP4)
                .Case("tq1_0", TQ1)
                .Case("tq2_0", TQ2)
                .Case("iq4_nl", IQ4NL)
                .Case("iq2_xxs", IQ2XXS)
                .Case("iq2_xs", IQ2XS)
                .Case("iq2_s", IQ2S)
                .Case("iq3_xxs", IQ3XXS)
                .Case("iq3_s", IQ3S)
                .Case("iq1_s", IQ1S)
                .Case("iq1_m", IQ1M)
                .Case("iq4_xs", IQ4XS)
                .Default(NONE);
  // Plain failure (NO ops emitted yet) so the caller falls back to the legacy
  // q4_0..q8_0 chain for the formats this function does not own.
  if (fmt == NONE)
    return mlir::failure();

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
  mlir::Type boolType = rewriter.getI1Type();
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
  mlir::Type i8PtrType = emitc::PointerType::get(constI8Type);
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
  };
  auto intLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, intType, std::to_string(v));
  };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType, std::to_string(v));
  };
  auto floatLit = [&](llvm::StringRef s) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, floatType, s);
  };
  auto addSz = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::AddOp>(loc, sizeType, a, b).getResult();
  };
  auto mulSz = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  // Integer-domain arithmetic (all in C `int`, matching ggml's promoted decode).
  auto iAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseAndOp>(loc, intType, a, b).getResult();
  };
  auto iOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseOrOp>(loc, intType, a, b).getResult();
  };
  auto iShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, intType, a, b)
        .getResult();
  };
  auto iShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, a, b)
        .getResult();
  };
  auto iSub = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::SubOp>(loc, intType, a, b).getResult();
  };
  auto iAdd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::AddOp>(loc, intType, a, b).getResult();
  };
  auto fMul = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::MulOp>(loc, floatType, a, b).getResult();
  };
  auto fSub = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::SubOp>(loc, floatType, a, b).getResult();
  };
  auto i2f = [&](mlir::Value v) -> mlir::Value {
    return rewriter.create<emitc::CastOp>(loc, floatType, v).getResult();
  };
  auto i2sz = [&](mlir::Value v) -> mlir::Value {
    return rewriter.create<emitc::CastOp>(loc, sizeType, v).getResult();
  };
  auto sz2i = [&](mlir::Value v) -> mlir::Value {
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };
  // Load *(base + off) (a byte cursor advanced by the size_t `off`) and widen to
  // `int`: a const uint8_t elem zero-extends, a const int8_t elem sign-extends
  // (the proven pointer-advance + subscript[0] idiom, no runtime subscript).
  auto loadIntAt = [&](mlir::Value base, mlir::Type basePtrTy, mlir::Value off,
                       mlir::Type elemTy) -> mlir::Value {
    mlir::Value p =
        rewriter.create<emitc::AddOp>(loc, basePtrTy, base, off).getResult();
    mlir::Value elem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(p),
                idxLit(0))
            .getResult();
    mlir::Value v =
        rewriter.create<emitc::LoadOp>(loc, elemTy, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };
  // *(ybFloat + off) = value  (store one f32 through the float* row cursor).
  auto storeF32At = [&](mlir::Value ybFloat, mlir::Value off,
                        mlir::Value value) {
    mlir::Value p =
        rewriter.create<emitc::AddOp>(loc, floatPtrType, ybFloat, off)
            .getResult();
    mlir::Value elem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(p),
                idxLit(0))
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, elem, value);
  };
  auto fp16ReadAt = [&](mlir::Value xb, int64_t off) -> mlir::Value {
    mlir::Value addr =
        off == 0 ? xb
                 : rewriter.create<emitc::AddOp>(loc, inputPtrType, xb,
                                                 sizeLit(off))
                       .getResult();
    return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                          mlir::ValueRange{addr}, opName, role,
                          llvm::StringRef("fcvt.s.h"));
  };
  // kvalues[idx] scalar codebook gather: the table is a function-local static
  // decl; index it via the same pointer-advance + subscript[0] idiom, then
  // sign-extend the int8 entry to int.
  auto codebookLoad = [&](llvm::StringRef tableName,
                          mlir::Value idxInt) -> mlir::Value {
    mlir::Value tbl =
        rewriter.create<emitc::LiteralOp>(loc, i8PtrType, tableName)
            .getResult();
    return loadIntAt(tbl, i8PtrType, i2sz(idxInt), constI8Type);
  };
  auto emitCodebookDecl = [&](llvm::StringRef name,
                              llvm::ArrayRef<int> entries) {
    std::string decl = "static const int8_t " + name.str() + "[16] = {";
    for (size_t i = 0; i < entries.size(); ++i) {
      if (i)
        decl += ", ";
      decl += std::to_string(entries[i]);
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);
  };
  // The nvfp4 per-sub-block UE4M3 fp8 -> fp32 HALF scale (ggml_ue4m3_to_fp32,
  // ggml-impl.h): e==0||e==0x7F -> 0; exp=(e>>3)&0xF, man=e&7; raw = exp==0 ?
  // ldexpf(man,-9) : ldexpf(1+man/8, exp-7); result = raw*0.5f. Structured emitc.
  auto ue4m3ScaleAt = [&](mlir::Value xb, mlir::Value off) -> mlir::Value {
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "ue4m3_scale"));
    mlir::Value p =
        rewriter.create<emitc::AddOp>(loc, u8PtrType, xb, off).getResult();
    mlir::Value elem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(p),
                idxLit(0))
            .getResult();
    mlir::Value e =
        rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    mlir::Value e32 =
        rewriter.create<emitc::CastOp>(loc, uintType, e).getResult();
    auto uLit = [&](llvm::StringRef s) {
      return rewriter.create<emitc::LiteralOp>(loc, uintType, s).getResult();
    };
    mlir::Value expShift =
        rewriter
            .create<emitc::BitwiseRightShiftOp>(loc, uintType, e32, uLit("3"))
            .getResult();
    mlir::Value expU =
        rewriter
            .create<emitc::BitwiseAndOp>(loc, uintType, expShift, uLit("0xF"))
            .getResult();
    mlir::Value manU =
        rewriter
            .create<emitc::BitwiseAndOp>(loc, uintType, e32, uLit("0x7"))
            .getResult();
    mlir::Value expInt =
        rewriter.create<emitc::CastOp>(loc, intType, expU).getResult();
    mlir::Value manFloat =
        i2f(rewriter.create<emitc::CastOp>(loc, intType, manU).getResult());
    mlir::Value denormRaw =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                         "ldexpf",
                                         mlir::ValueRange{manFloat, intLit(-9)})
            .getResult(0);
    mlir::Value normMant = rewriter.create<emitc::AddOp>(
        loc, floatType, floatLit("1.0f"),
        rewriter.create<emitc::DivOp>(loc, floatType, manFloat,
                                      floatLit("8.0f")));
    mlir::Value normExp =
        rewriter.create<emitc::SubOp>(loc, intType, expInt, intLit(7));
    mlir::Value normRaw =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                         "ldexpf",
                                         mlir::ValueRange{normMant, normExp})
            .getResult(0);
    mlir::Value isDenorm =
        rewriter
            .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::eq, expU,
                                  uLit("0"))
            .getResult();
    mlir::Value raw =
        rewriter
            .create<emitc::ConditionalOp>(loc, floatType, isDenorm, denormRaw,
                                          normRaw)
            .getResult();
    mlir::Value scaled = fMul(raw, floatLit("0.5f"));
    mlir::Value isZero =
        rewriter
            .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::eq, e32,
                                  uLit("0"))
            .getResult();
    mlir::Value isSpec =
        rewriter
            .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::eq, e32,
                                  uLit("0x7F"))
            .getResult();
    mlir::Value special =
        rewriter.create<emitc::LogicalOrOp>(loc, boolType, isZero, isSpec)
            .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, floatType, special,
                                      floatLit("0.0f"), scaled)
        .getResult();
  };

  // --- IQ grid/signs decode helpers (shared by the 8 IQ grid-table formats;
  // each reuses the SAME canonical grid/signs table decl the block-dot vec_dot
  // lowering emits, then a scalar AoS super-block loop byte-exact to ggml's
  // reference dequantize_row_iq*). [L-6] wiring != construction. ---
  mlir::Type i64PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
  mlir::Type u32PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
  mlir::Type u64PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint64_t"));
  auto iMul = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::MulOp>(loc, intType, a, b).getResult();
  };
  auto fAdd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::AddOp>(loc, floatType, a, b).getResult();
  };
  auto uLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, uintType,
                                             std::to_string(v) + "u");
  };
  auto uAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseAndOp>(loc, uintType, a, b).getResult();
  };
  auto uOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseOrOp>(loc, uintType, a, b).getResult();
  };
  auto uShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uintType, a, b)
        .getResult();
  };
  auto uShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseRightShiftOp>(loc, uintType, a, b)
        .getResult();
  };
  auto u2i = [&](mlir::Value v) -> mlir::Value {
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };
  // Load one uint8 byte at (base + off) and ZERO-extend to uint32_t.
  auto loadU8AsUint = [&](mlir::Value base, mlir::Type basePtrTy,
                          mlir::Value off) -> mlir::Value {
    mlir::Value p =
        rewriter.create<emitc::AddOp>(loc, basePtrTy, base, off).getResult();
    mlir::Value elem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(p),
                idxLit(0))
            .getResult();
    mlir::Value v =
        rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, uintType, v).getResult();
  };
  // uint16 assembled little-endian from 2 bytes at (base + off) (ggml's memcpy /
  // aligned uint16 read; assembled to be alignment/endianness-safe).
  auto u16ReadAt = [&](mlir::Value base, mlir::Type basePtrTy,
                       mlir::Value off) -> mlir::Value {
    mlir::Value b0 = loadU8AsUint(base, basePtrTy, off);
    mlir::Value b1 = loadU8AsUint(base, basePtrTy, addSz(off, sizeLit(1)));
    return uOr(b0, uShl(b1, uLit(8)));
  };
  // uint32 assembled little-endian from 4 bytes at (base + off) (ggml's memcpy
  // into aux32; assembled to be alignment/endianness-safe).
  auto u32ReadAt = [&](mlir::Value base, mlir::Type basePtrTy,
                       mlir::Value off) -> mlir::Value {
    mlir::Value b0 = loadU8AsUint(base, basePtrTy, off);
    mlir::Value b1 = loadU8AsUint(base, basePtrTy, addSz(off, sizeLit(1)));
    mlir::Value b2 = loadU8AsUint(base, basePtrTy, addSz(off, sizeLit(2)));
    mlir::Value b3 = loadU8AsUint(base, basePtrTy, addSz(off, sizeLit(3)));
    return uOr(uOr(uOr(b0, uShl(b1, uLit(8))), uShl(b2, uLit(16))),
               uShl(b3, uLit(24)));
  };
  // A `(const int8_t *)<gridArray>` byte view of a non-int8 grid table (int64 /
  // uint32 / uint64), created via a name literal + emitc pointer cast -- exactly
  // ggml's `(const uint8_t *)(grid + idx)` byte read (all grid bytes < 128 so the
  // int8 sign-extending read equals ggml's uint8 read).
  auto castI8Base = [&](llvm::StringRef nm, mlir::Type declPtrTy) -> mlir::Value {
    mlir::Value lit =
        rewriter.create<emitc::LiteralOp>(loc, declPtrTy, nm).getResult();
    return rewriter.create<emitc::CastOp>(loc, i8PtrType, lit).getResult();
  };
  // A name literal at a given pointer type (int8/uint8 tables decay directly).
  auto plainBase = [&](llvm::StringRef nm, mlir::Type ptrTy) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, ptrTy, nm).getResult();
  };
  // Read grid/signs byte (sign-extended int8 -> int) at flat int index fi.
  auto tableByteAtInt = [&](mlir::Value i8base,
                            mlir::Value fiInt) -> mlir::Value {
    return loadIntAt(i8base, i8PtrType, i2sz(fiInt), constI8Type);
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // ggml FP4 e2m1 / iq4_nl non-linear codebooks (ggml-common.h). Emitted ONCE as
  // function-local static decls above the super-block loop.
  static const int kvaluesMxfp4[16] = {0, 1, 2, 3,  4,  6,  8,  12,
                                       0, -1, -2, -3, -4, -6, -8, -12};
  static const int kvaluesIq4nl[16] = {-127, -104, -83, -65, -49, -35,
                                        -22,  -10,  1,   13,  25,  38,
                                        53,   69,   89,  113};
  llvm::StringRef codebookName;
  if (fmt == MXFP4) {
    codebookName = "tcrv_dequant_mxfp4_kvalues";
    emitCodebookDecl(codebookName, kvaluesMxfp4);
  } else if (fmt == NVFP4) {
    codebookName = "tcrv_dequant_nvfp4_kvalues";
    emitCodebookDecl(codebookName, kvaluesMxfp4);
  } else if (fmt == IQ4NL) {
    codebookName = "tcrv_dequant_iq4nl_kvalues";
    emitCodebookDecl(codebookName, kvaluesIq4nl);
  }
  if (fmt == TQ1) {
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_dequant_tq1_0_pow3[6] = "
             "{1, 3, 9, 27, 81, 243};");
  }

  // The IQ grid/signs codebook table decls (function-local statics), emitted from
  // the SAME canonical grid/signs anchors the block-dot vec_dot lowerings render
  // (byte-identical). iq4_xs reuses the 16-entry iq4_nl codebook.
  switch (fmt) {
  case IQ2XXS:
    emitIQ2XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XXSCanonicalSigns64TableDecl(rewriter, loc);
    break;
  case IQ2XS:
    emitIQ2XSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XSCanonicalSigns64TableDecl(rewriter, loc);
    break;
  case IQ2S:
    emitIQ2SCanonicalGridTableDecl(rewriter, loc);
    emitIQ2SCanonicalSigns256TableDecl(rewriter, loc);
    break;
  case IQ3XXS:
    emitIQ3XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ3XXSCanonicalKsignsTableDecl(rewriter, loc);
    break;
  case IQ3S:
    emitIQ3SCanonicalGridTableDecl(rewriter, loc);
    break;
  case IQ1S:
    emitIQ1SCanonicalGridTableDecl(rewriter, loc);
    break;
  case IQ1M:
    emitIQ1MCanonicalGridTableDecl(rewriter, loc);
    break;
  case IQ4XS:
    codebookName = "tcrv_dequant_iq4nl_kvalues";
    emitCodebookDecl(codebookName, kvaluesIq4nl);
    break;
  default: break;
  }

  // Per-format super-block geometry (element_count = k; qk = elements/super-block,
  // stride = super-block byte size). nb = k / qk. All IQ K-quant super-blocks are
  // QK_K = 256 (qk default); strides are the ggml block_iqX AoS byte sizes.
  int64_t qk = 256, stride = 0;
  switch (fmt) {
  case Q2K: stride = 84; break;
  case Q3K: stride = 110; break;
  case Q4K: stride = 144; break;
  case Q5K: stride = 176; break;
  case Q6K: stride = 210; break;
  case MXFP4: qk = 32; stride = 17; break;
  case NVFP4: qk = 64; stride = 36; break;
  case TQ1: stride = 54; break;
  case TQ2: stride = 66; break;
  case IQ4NL: qk = 32; stride = 18; break;
  case IQ2XXS: stride = 66; break;
  case IQ2XS: stride = 74; break;
  case IQ2S: stride = 82; break;
  case IQ3XXS: stride = 98; break;
  case IQ3S: stride = 110; break;
  case IQ1S: stride = 50; break;
  case IQ1M: stride = 56; break;
  case IQ4XS: stride = 136; break;
  default: break;
  }

  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1),
                                                /*bodyBuilder=*/nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    // const uint8_t *xb = x + ib*stride;   float *yb = y + ib*qk;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter
                         .create<emitc::AddOp>(loc, inputPtrType, input,
                                               mulSz(ib, sizeLit(stride)))
                         .getResult();
    mlir::Value ybRaw = rewriter
                            .create<emitc::AddOp>(loc, outputPtrType, output,
                                                  mulSz(ib, sizeLit(qk)))
                            .getResult();
    mlir::Value yb =
        rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    // A small helper: emit an inner element loop `for (size_t v = 0; v < n; ++v)`.
    auto elemLoop = [&](int64_t n,
                        llvm::function_ref<void(mlir::Value)> body) {
      auto f = rewriter.create<emitc::ForOp>(loc, sizeLit(0), sizeLit(n),
                                             sizeLit(1), /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(f.getBody());
      body(f.getInductionVar());
    };

    if (fmt == IQ4NL || fmt == MXFP4) {
      // qk=32 nibble codebook: y[j] = d*kv[qs[j]&0xF]; y[j+16] = d*kv[qs[j]>>4].
      int64_t dOff = 0, qsOff = 2;
      mlir::Value d;
      if (fmt == MXFP4) {
        qsOff = 1;
        d = emitE8M0HalfScale(rewriter, loc, xb, opName, role);
      } else {
        d = fp16ReadAt(xb, dOff);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "codebook_nibble_decode"));
      elemLoop(qk / 2, [&](mlir::Value j) {
        mlir::Value qsAddr = addSz(sizeLit(qsOff), j);
        mlir::Value qi = loadIntAt(xb, inputPtrType, qsAddr, constU8Type);
        mlir::Value nlo = iAnd(qi, intLit(0x0F));
        mlir::Value nhi = iShr(qi, intLit(4));
        mlir::Value v0 = i2f(codebookLoad(codebookName, nlo));
        mlir::Value v1 = i2f(codebookLoad(codebookName, nhi));
        storeF32At(yb, j, fMul(v0, d));
        storeF32At(yb, addSz(j, sizeLit(qk / 2)), fMul(v1, d));
      });
    } else if (fmt == NVFP4) {
      // Four 16-element UE4M3-scaled sub-blocks; codebook nibble split per sub.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "nvfp4_sub_decode"));
      for (int64_t s = 0; s < 4; ++s) {
        mlir::Value d = ue4m3ScaleAt(xb, sizeLit(s)); // d[s] at offset s
        int64_t subOut = s * 16;                      // yb + s*16
        int64_t subQs = 4 + s * 8;                    // qs at +4, 8 bytes/sub
        elemLoop(8, [&](mlir::Value j) {
          mlir::Value qi =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(subQs), j), constU8Type);
          mlir::Value v0 = i2f(codebookLoad(codebookName, iAnd(qi, intLit(0x0F))));
          mlir::Value v1 = i2f(codebookLoad(codebookName, iShr(qi, intLit(4))));
          storeF32At(yb, addSz(sizeLit(subOut), j), fMul(v0, d));
          storeF32At(yb, addSz(sizeLit(subOut + 8), j), fMul(v1, d));
        });
      }
    } else if (fmt == TQ2) {
      // 2-bit ternary: q=(qs[j+m]>>(2l))&3; y=(q-1)*d. j in {0,32}, l 0..4, m 0..32.
      mlir::Value d = fp16ReadAt(xb, 64);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "tq2_0_decode"));
      for (int64_t jb = 0; jb < 2; ++jb) {
        for (int64_t l = 0; l < 4; ++l) {
          int64_t outBase = jb * 128 + l * 32;
          int64_t qsBase = jb * 32;
          elemLoop(32, [&](mlir::Value m) {
            mlir::Value qi = loadIntAt(xb, inputPtrType,
                                       addSz(sizeLit(qsBase), m), constU8Type);
            mlir::Value q = iAnd(iShr(qi, intLit(2 * l)), intLit(3));
            mlir::Value val = fMul(i2f(iSub(q, intLit(1))), d);
            storeF32At(yb, addSz(sizeLit(outBase), m), val);
          });
        }
      }
    } else if (fmt == TQ1) {
      // 1.6-bit ternary base-3: q=qs*pow3[n] (mod 256); xi=((q*3)>>8); y=(xi-1)*d.
      mlir::Value d = fp16ReadAt(xb, 52);
      llvm::StringRef pow3 = "tcrv_dequant_tq1_0_pow3";
      mlir::Value pow3Base =
          rewriter.create<emitc::LiteralOp>(loc, u8PtrType, pow3).getResult();
      // xi = ((uint8_t)(qs*pow3n) * 3) >> 8; store (xi-1)*d at outIdx.
      auto tq1Decode = [&](mlir::Value qbyteInt, mlir::Value pow3n,
                           mlir::Value outIdx) {
        mlir::Value prod = rewriter
                               .create<emitc::MulOp>(loc, intType, qbyteInt,
                                                     pow3n)
                               .getResult();
        mlir::Value qmod = iAnd(prod, intLit(0xFF)); // uint8_t truncation
        mlir::Value xi = iShr(rewriter.create<emitc::MulOp>(loc, intType, qmod,
                                                            intLit(3))
                                  .getResult(),
                              intLit(8));
        storeF32At(yb, outIdx, fMul(i2f(iSub(xi, intLit(1))), d));
      };
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "tq1_0_qs_full"));
      // Section 1: qs[0..31], n 0..4, m 0..31 -> out n*32+m.
      for (int64_t n = 0; n < 5; ++n) {
        mlir::Value pow3n = loadIntAt(pow3Base, u8PtrType, sizeLit(n),
                                      constU8Type);
        int64_t base = n * 32;
        elemLoop(32, [&](mlir::Value m) {
          mlir::Value qb = loadIntAt(xb, inputPtrType, m, constU8Type);
          tq1Decode(qb, pow3n, addSz(sizeLit(base), m));
        });
      }
      // Section 2: qs[32..47], n 0..4, m 0..15 -> out 160 + n*16+m.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "tq1_0_qs_tail"));
      for (int64_t n = 0; n < 5; ++n) {
        mlir::Value pow3n = loadIntAt(pow3Base, u8PtrType, sizeLit(n),
                                      constU8Type);
        int64_t base = 160 + n * 16;
        elemLoop(16, [&](mlir::Value m) {
          mlir::Value qb =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(32), m), constU8Type);
          tq1Decode(qb, pow3n, addSz(sizeLit(base), m));
        });
      }
      // Section 3: qh[0..3] at +48, n 0..3, j 0..3 -> out 240 + n*4+j.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "tq1_0_qh"));
      for (int64_t n = 0; n < 4; ++n) {
        mlir::Value pow3n = loadIntAt(pow3Base, u8PtrType, sizeLit(n),
                                      constU8Type);
        int64_t base = 240 + n * 4;
        elemLoop(4, [&](mlir::Value j) {
          mlir::Value qb =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(48), j), constU8Type);
          tq1Decode(qb, pow3n, addSz(sizeLit(base), j));
        });
      }
    } else if (fmt == Q2K) {
      // scales[16]@0, qs[64]@16, d@80, dmin@82. y = dl*((q>>shift)&3) - ml.
      mlir::Value d = fp16ReadAt(xb, 80);
      mlir::Value dmin = fp16ReadAt(xb, 82);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "q2_K_decode"));
      elemLoop(2, [&](mlir::Value nn) {
        mlir::Value nn8 = mulSz(nn, sizeLit(8));
        mlir::Value qBlk = addSz(sizeLit(16), mulSz(nn, sizeLit(32)));
        mlir::Value outBlk = mulSz(nn, sizeLit(128));
        elemLoop(4, [&](mlir::Value j) {
          mlir::Value shift = iShl(sz2i(j), intLit(1)); // 2*j
          mlir::Value is0 = addSz(nn8, mulSz(j, sizeLit(2)));
          mlir::Value outJ = addSz(outBlk, mulSz(j, sizeLit(32)));
          for (int64_t half = 0; half < 2; ++half) {
            mlir::Value sc =
                loadIntAt(xb, inputPtrType, addSz(is0, sizeLit(half)),
                          constU8Type);
            mlir::Value dl = fMul(d, i2f(iAnd(sc, intLit(0xF))));
            mlir::Value ml = fMul(dmin, i2f(iShr(sc, intLit(4))));
            mlir::Value qHalf = addSz(qBlk, sizeLit(half * 16));
            mlir::Value outHalf = addSz(outJ, sizeLit(half * 16));
            elemLoop(16, [&](mlir::Value l) {
              mlir::Value q =
                  loadIntAt(xb, inputPtrType, addSz(qHalf, l), constU8Type);
              mlir::Value qv = iAnd(iShr(q, shift), intLit(3));
              storeF32At(yb, addSz(outHalf, l), fSub(fMul(dl, i2f(qv)), ml));
            });
          }
        });
      });
    } else if (fmt == Q3K) {
      // hmask[32]@0, qs[64]@32, scales[12]@96, d@108; 6-bit signed scales via aux.
      mlir::Value dAll = fp16ReadAt(xb, 108);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "q3_K_decode"));
      // The derived 6-bit scale for compile-time index `is` (0..15), from the 12
      // packed scale bytes at +96 (ggml's aux kmask1/kmask2 shuffle, byte-exact).
      auto q3Scale = [&](int is) -> mlir::Value {
        int group = is / 4, b = is % 4;
        auto sb = [&](int k) {
          return loadIntAt(xb, inputPtrType, sizeLit(96 + k), constU8Type);
        };
        mlir::Value lowNib, hiPart;
        if (group == 0) {
          lowNib = iAnd(sb(b), intLit(0x0F));
          hiPart = iShl(iAnd(sb(8 + b), intLit(0x03)), intLit(4));
        } else if (group == 1) {
          lowNib = iAnd(sb(4 + b), intLit(0x0F));
          hiPart = iShl(iAnd(iShr(sb(8 + b), intLit(2)), intLit(0x03)),
                        intLit(4));
        } else if (group == 2) {
          lowNib = iAnd(iShr(sb(b), intLit(4)), intLit(0x0F));
          hiPart = iShl(iAnd(iShr(sb(8 + b), intLit(4)), intLit(0x03)),
                        intLit(4));
        } else {
          lowNib = iAnd(iShr(sb(4 + b), intLit(4)), intLit(0x0F));
          hiPart = iShl(iAnd(iShr(sb(8 + b), intLit(6)), intLit(0x03)),
                        intLit(4));
        }
        return iOr(lowNib, hiPart);
      };
      for (int64_t nn = 0; nn < 2; ++nn) {
        int64_t qBlk = 32 + nn * 32; // qs@32, +32 per n
        int64_t outBlk = nn * 128;
        for (int64_t j = 0; j < 4; ++j) {
          int64_t shift = 2 * j;
          int64_t mbit = nn * 4 + j; // hmask bit tested for the high term
          int64_t outJ = outBlk + j * 32;
          for (int64_t half = 0; half < 2; ++half) {
            int is = (int)(nn * 8 + 2 * j + half);
            mlir::Value dl = fMul(dAll, i2f(iSub(q3Scale(is), intLit(32))));
            int64_t qHalf = qBlk + half * 16;   // q[l] / q[l+16]
            int64_t hmHalf = half * 16;         // hmask[l] / hmask[l+16]
            int64_t outHalf = outJ + half * 16;
            elemLoop(16, [&](mlir::Value l) {
              mlir::Value q = loadIntAt(xb, inputPtrType,
                                        addSz(sizeLit(qHalf), l), constU8Type);
              mlir::Value hm = loadIntAt(xb, inputPtrType,
                                         addSz(sizeLit(hmHalf), l), constU8Type);
              mlir::Value qbits = iAnd(iShr(q, intLit(shift)), intLit(3));
              // (hm & (1<<mbit)) ? 0 : 4  ==  (1 - bit) << 2, bit in {0,1}.
              mlir::Value bit = iAnd(iShr(hm, intLit(mbit)), intLit(1));
              mlir::Value term = iShl(iSub(intLit(1), bit), intLit(2));
              mlir::Value qdec = iSub(qbits, term);
              storeF32At(yb, addSz(sizeLit(outHalf), l), fMul(dl, i2f(qdec)));
            });
          }
        }
      }
    } else if (fmt == Q4K || fmt == Q5K) {
      // d@0, dmin@2, scales[12]@4; q4_K qs@16; q5_K qh@16, qs@48. get_scale_min_k4.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value dmin = fp16ReadAt(xb, 2);
      bool isQ5 = (fmt == Q5K);
      int64_t qsOff = isQ5 ? 48 : 16;
      rewriter.create<emitc::VerbatimOp>(
          loc,
          stepComment(opName, role, isQ5 ? "q5_K_decode" : "q4_K_decode"));
      // get_scale_min_k4(j) for compile-time j (0..7), scales at +4.
      auto scaleMin = [&](int j, mlir::Value &scOut, mlir::Value &mOut) {
        auto sc = [&](int k) {
          return loadIntAt(xb, inputPtrType, sizeLit(4 + k), constU8Type);
        };
        if (j < 4) {
          scOut = iAnd(sc(j), intLit(63));
          mOut = iAnd(sc(j + 4), intLit(63));
        } else {
          scOut = iOr(iAnd(sc(j + 4), intLit(0x0F)),
                      iShl(iShr(sc(j - 4), intLit(6)), intLit(4)));
          mOut = iOr(iShr(sc(j + 4), intLit(4)),
                     iShl(iShr(sc(j), intLit(6)), intLit(4)));
        }
      };
      for (int64_t jj = 0; jj < 4; ++jj) {
        mlir::Value sc0, m0, sc1, m1v;
        scaleMin((int)(2 * jj), sc0, m0);
        mlir::Value d1 = fMul(d, i2f(sc0));
        mlir::Value ml1 = fMul(dmin, i2f(m0));
        scaleMin((int)(2 * jj + 1), sc1, m1v);
        mlir::Value d2 = fMul(d, i2f(sc1));
        mlir::Value ml2 = fMul(dmin, i2f(m1v));
        int64_t qBlk = qsOff + jj * 32; // ql advances 32/super-sub
        int64_t outBlk = jj * 64;
        // low half (32): d1*((ql&0xF)[+qh bit]) - ml1
        elemLoop(32, [&](mlir::Value l) {
          mlir::Value ql =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(qBlk), l), constU8Type);
          mlir::Value v = iAnd(ql, intLit(0x0F));
          if (isQ5) {
            mlir::Value qh = loadIntAt(xb, inputPtrType, addSz(sizeLit(16), l),
                                       constU8Type); // qh@16
            mlir::Value hb =
                iShl(iAnd(iShr(qh, intLit((int)(2 * jj))), intLit(1)), intLit(4));
            v = iAdd(v, hb);
          }
          storeF32At(yb, addSz(sizeLit(outBlk), l), fSub(fMul(d1, i2f(v)), ml1));
        });
        // high half (32): d2*((ql>>4)[+qh bit]) - ml2
        elemLoop(32, [&](mlir::Value l) {
          mlir::Value ql =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(qBlk), l), constU8Type);
          mlir::Value v = iShr(ql, intLit(4));
          if (isQ5) {
            mlir::Value qh = loadIntAt(xb, inputPtrType, addSz(sizeLit(16), l),
                                       constU8Type);
            mlir::Value hb = iShl(
                iAnd(iShr(qh, intLit((int)(2 * jj + 1))), intLit(1)), intLit(4));
            v = iAdd(v, hb);
          }
          storeF32At(yb, addSz(sizeLit(outBlk + 32), l),
                     fSub(fMul(d2, i2f(v)), ml2));
        });
      }
    } else if (fmt == Q6K) {
      // ql[128]@0, qh[64]@128, scales[16]@192 (int8), d@208.
      mlir::Value d = fp16ReadAt(xb, 208);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "q6_K_decode"));
      mlir::Value xbI8 =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, xb).getResult();
      for (int64_t nn = 0; nn < 2; ++nn) {
        int64_t qlBlk = nn * 64;    // ql@0, +64
        int64_t qhBlk = 128 + nn * 32; // qh@128, +32
        int64_t scBlk = 192 + nn * 8;  // scales@192 (int8), +8
        int64_t outBlk = nn * 128;
        elemLoop(32, [&](mlir::Value l) {
          mlir::Value is =
              rewriter.create<emitc::DivOp>(loc, sizeType, l, sizeLit(16))
                  .getResult(); // l/16 in {0,1}
          mlir::Value ql0 =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(qlBlk), l), constU8Type);
          mlir::Value ql32 = loadIntAt(xb, inputPtrType,
                                       addSz(sizeLit(qlBlk + 32), l),
                                       constU8Type);
          mlir::Value qh =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(qhBlk), l), constU8Type);
          // q1..q4 = ((low|high2<<4)) - 32 (all values are 0..63, so int8-safe).
          mlir::Value q1 =
              iSub(iOr(iAnd(ql0, intLit(0x0F)),
                       iShl(iAnd(iShr(qh, intLit(0)), intLit(3)), intLit(4))),
                   intLit(32));
          mlir::Value q2 =
              iSub(iOr(iAnd(ql32, intLit(0x0F)),
                       iShl(iAnd(iShr(qh, intLit(2)), intLit(3)), intLit(4))),
                   intLit(32));
          mlir::Value q3 =
              iSub(iOr(iShr(ql0, intLit(4)),
                       iShl(iAnd(iShr(qh, intLit(4)), intLit(3)), intLit(4))),
                   intLit(32));
          mlir::Value q4 =
              iSub(iOr(iShr(ql32, intLit(4)),
                       iShl(iAnd(iShr(qh, intLit(6)), intLit(3)), intLit(4))),
                   intLit(32));
          auto scLd = [&](int64_t k) {
            // scales[is + k], scales int8 at byte offset scBlk (= 192 + nn*8).
            return loadIntAt(xbI8, i8PtrType, addSz(sizeLit(scBlk + k), is),
                             constI8Type);
          };
          // y[l], y[l+32], y[l+64], y[l+96] = d * sc[is+{0,2,4,6}] * q{1..4}.
          mlir::Value o = addSz(sizeLit(outBlk), l);
          storeF32At(yb, o, fMul(fMul(d, i2f(scLd(0))), i2f(q1)));
          storeF32At(yb, addSz(o, sizeLit(32)),
                     fMul(fMul(d, i2f(scLd(2))), i2f(q2)));
          storeF32At(yb, addSz(o, sizeLit(64)),
                     fMul(fMul(d, i2f(scLd(4))), i2f(q3)));
          storeF32At(yb, addSz(o, sizeLit(96)),
                     fMul(fMul(d, i2f(scLd(6))), i2f(q4)));
        });
      }
    } else if (fmt == IQ4XS) {
      // d@0, scales_h(u16)@2, scales_l[4]@4, qs[128]@8. Per ib (8 sub-blocks):
      // 6-bit ls from a scales_l nibble + a scales_h 2-bit; dl = d*(ls-32); the
      // 16-entry iq4_nl codebook nibble decode (byte-exact dequantize_row_iq4_xs).
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value sh = u16ReadAt(xb, inputPtrType, sizeLit(2)); // scales_h once
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq4_xs_decode"));
      for (int64_t ib = 0; ib < 8; ++ib) {
        mlir::Value scl =
            loadIntAt(xb, inputPtrType, sizeLit(4 + ib / 2), constU8Type);
        mlir::Value low = iAnd(iShr(scl, intLit(4 * (ib % 2))), intLit(0xF));
        mlir::Value hi =
            iShl(u2i(uAnd(uShr(sh, uLit(2 * ib)), uLit(3))), intLit(4));
        mlir::Value dl = fMul(d, i2f(iSub(iOr(low, hi), intLit(32))));
        int64_t qsBase = 8 + ib * 16, outBase = ib * 32;
        elemLoop(16, [&](mlir::Value j) {
          mlir::Value qb =
              loadIntAt(xb, inputPtrType, addSz(sizeLit(qsBase), j),
                        constU8Type);
          mlir::Value lo = i2f(codebookLoad(codebookName, iAnd(qb, intLit(0xF))));
          mlir::Value hg = i2f(codebookLoad(codebookName, iShr(qb, intLit(4))));
          storeF32At(yb, addSz(sizeLit(outBase), j), fMul(dl, lo));
          storeF32At(yb, addSz(sizeLit(outBase + 16), j), fMul(dl, hg));
        });
      }
    } else if (fmt == IQ1S) {
      // d@0, qs[32]@2, qh(u16)[8]@34. iq1s_grid ternary (int8), delta = +-0.125.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value grid = castI8Base("tcrv_iq1s_grid", u64PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq1_s_decode"));
      for (int64_t ib = 0; ib < 8; ++ib) {
        mlir::Value qhv = u16ReadAt(xb, inputPtrType, sizeLit(34 + ib * 2));
        // dl = d * (2*((qh>>12)&7) + 1).
        mlir::Value g3 = u2i(uAnd(uShr(qhv, uLit(12)), uLit(7)));
        mlir::Value dl =
            fMul(d, i2f(iAdd(iMul(intLit(2), g3), intLit(1))));
        // delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA.
        mlir::Value dbit = u2i(uShr(uAnd(qhv, uLit(0x8000)), uLit(15)));
        mlir::Value delta = fMul(i2f(iSub(intLit(1), iMul(intLit(2), dbit))),
                                 floatLit("0.125f"));
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value qsb = loadIntAt(xb, inputPtrType,
                                      sizeLit(2 + ib * 4 + l), constU8Type);
          // idx = qs[l] | (((qh >> 3l) & 7) << 8).
          mlir::Value hb =
              iShl(u2i(uAnd(uShr(qhv, uLit(3 * l)), uLit(7))), intLit(8));
          mlir::Value base8 = iMul(iOr(qsb, hb), intLit(8));
          int64_t outBase = ib * 32 + l * 8;
          elemLoop(8, [&](mlir::Value j) {
            mlir::Value g = tableByteAtInt(grid, iAdd(base8, sz2i(j)));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(dl, fAdd(i2f(g), delta)));
          });
        }
      }
    } else if (fmt == IQ1M) {
      // qs[32]@0, qh[16]@32, scales(u16)[4]@48. NO d field -- the super-block d is
      // the packed iq1m_scale fp16 reconstructed from the 4 scale words then read
      // AS _Float16 (bit reinterpret via a uint16 lvalue + address-of, the SAME
      // idiom the iq1_m block-dot vec_dot uses). iq1s_grid ternary, delta=+-0.125.
      mlir::Type u16Type = emitc::OpaqueType::get(ctx, "uint16_t");
      mlir::Type u16LValuePtrType = emitc::PointerType::get(u16Type);
      mlir::Value grid = castI8Base("tcrv_iq1m_grid", u64PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq1m_scale_reconstruct"));
      mlir::Value sc0 = u16ReadAt(xb, inputPtrType, sizeLit(48));
      mlir::Value sc1 = u16ReadAt(xb, inputPtrType, sizeLit(50));
      mlir::Value sc2 = u16ReadAt(xb, inputPtrType, sizeLit(52));
      mlir::Value sc3 = u16ReadAt(xb, inputPtrType, sizeLit(54));
      // scbits = (sc0>>12)|((sc1>>8)&0xf0)|((sc2>>4)&0xf00)|(sc3&0xf000).
      mlir::Value scbits =
          uOr(uOr(uOr(uShr(sc0, uLit(12)), uAnd(uShr(sc1, uLit(8)), uLit(0xf0))),
                  uAnd(uShr(sc2, uLit(4)), uLit(0xf00))),
              uAnd(sc3, uLit(0xf000)));
      auto scbitsVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(u16Type),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value scbitsU16 =
          rewriter.create<emitc::CastOp>(loc, u16Type, scbits).getResult();
      rewriter.create<emitc::AssignOp>(loc, scbitsVar, scbitsU16);
      mlir::Value scbitsAddr =
          rewriter
              .create<emitc::ApplyOp>(loc, u16LValuePtrType,
                                      rewriter.getStringAttr("&"), scbitsVar)
              .getResult();
      mlir::Value d = rewriter
                          .create<emitc::CallOpaqueOp>(
                              loc, mlir::TypeRange{floatType}, fp16ReadCallee,
                              mlir::ValueRange{scbitsAddr})
                          .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq1_m_decode"));
      mlir::Value scInt[4] = {u2i(sc0), u2i(sc1), u2i(sc2), u2i(sc3)};
      auto deltaOf = [&](mlir::Value qhByte, int64_t bit) -> mlir::Value {
        mlir::Value b = iAnd(iShr(qhByte, intLit(bit)), intLit(1));
        return fMul(i2f(iSub(intLit(1), iMul(intLit(2), b))),
                    floatLit("0.125f"));
      };
      for (int64_t ib = 0; ib < 8; ++ib) {
        mlir::Value scv = scInt[ib / 2];
        int64_t sh0 = 6 * (ib % 2) + 0, sh3 = 6 * (ib % 2) + 3;
        mlir::Value dl1 = fMul(
            d, i2f(iAdd(iMul(intLit(2), iAnd(iShr(scv, intLit(sh0)), intLit(7))),
                        intLit(1))));
        mlir::Value dl2 = fMul(
            d, i2f(iAdd(iMul(intLit(2), iAnd(iShr(scv, intLit(sh3)), intLit(7))),
                        intLit(1))));
        mlir::Value qh0 = loadIntAt(xb, inputPtrType, sizeLit(32 + 2 * ib + 0),
                                    constU8Type);
        mlir::Value qh1 = loadIntAt(xb, inputPtrType, sizeLit(32 + 2 * ib + 1),
                                    constU8Type);
        mlir::Value qsb[4] = {
            loadIntAt(xb, inputPtrType, sizeLit(4 * ib + 0), constU8Type),
            loadIntAt(xb, inputPtrType, sizeLit(4 * ib + 1), constU8Type),
            loadIntAt(xb, inputPtrType, sizeLit(4 * ib + 2), constU8Type),
            loadIntAt(xb, inputPtrType, sizeLit(4 * ib + 3), constU8Type)};
        // idx[l] = qs[l] | ((qh_hi << shift) & 0x700).
        mlir::Value idx[4] = {
            iOr(qsb[0], iAnd(iShl(qh0, intLit(8)), intLit(0x700))),
            iOr(qsb[1], iAnd(iShl(qh0, intLit(4)), intLit(0x700))),
            iOr(qsb[2], iAnd(iShl(qh1, intLit(8)), intLit(0x700))),
            iOr(qsb[3], iAnd(iShl(qh1, intLit(4)), intLit(0x700)))};
        mlir::Value delta[4] = {deltaOf(qh0, 3), deltaOf(qh0, 7),
                                deltaOf(qh1, 3), deltaOf(qh1, 7)};
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value dl = (l < 2) ? dl1 : dl2;
          mlir::Value base8 = iMul(idx[l], intLit(8));
          mlir::Value dlt = delta[l];
          int64_t outBase = ib * 32 + l * 8;
          elemLoop(8, [&](mlir::Value j) {
            mlir::Value g = tableByteAtInt(grid, iAdd(base8, sz2i(j)));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(dl, fAdd(i2f(g), dlt)));
          });
        }
      }
    } else if (fmt == IQ2XXS) {
      // d@0, qs(u16)[32]@2 (= aux). grid-of-8 (int64) + signs64 (+-1). The 8-byte
      // aux group per ib32: aux8[l] = grid index byte, aux32[1] = scale + sign
      // selectors. Byte-exact to dequantize_row_iq2_xxs.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value grid = castI8Base("tcrv_iq2xxs_grid", i64PtrType);
      mlir::Value signs = plainBase("tcrv_iq2xxs_signs64", i8PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq2_xxs_decode"));
      for (int64_t ib32 = 0; ib32 < 8; ++ib32) {
        int64_t auxBase = 2 + 8 * ib32;
        mlir::Value aux1 = u32ReadAt(xb, inputPtrType, sizeLit(auxBase + 4));
        // db = d * (0.5 + (aux1>>28)) * 0.25.
        mlir::Value db = fMul(fMul(d, fAdd(floatLit("0.5f"),
                                           i2f(u2i(uShr(aux1, uLit(28)))))),
                              floatLit("0.25f"));
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value a = loadIntAt(xb, inputPtrType, sizeLit(auxBase + l),
                                    constU8Type);
          mlir::Value sel = u2i(uAnd(uShr(aux1, uLit(7 * l)), uLit(127)));
          mlir::Value gBase = iMul(a, intLit(8));
          mlir::Value sBase = iMul(sel, intLit(8));
          int64_t outBase = ib32 * 32 + l * 8;
          elemLoop(8, [&](mlir::Value j) {
            mlir::Value ji = sz2i(j);
            mlir::Value g = tableByteAtInt(grid, iAdd(gBase, ji));
            mlir::Value sg = tableByteAtInt(signs, iAdd(sBase, ji));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(fMul(db, i2f(g)), i2f(sg)));
          });
        }
      }
    } else if (fmt == IQ2XS) {
      // d@0, qs(u16)[32]@2, scales[8]@66. grid-512 (int64) + signs64. Byte-exact
      // to dequantize_row_iq2_xs.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value grid = castI8Base("tcrv_iq2xs_grid", i64PtrType);
      mlir::Value signs = plainBase("tcrv_iq2xs_signs64", i8PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq2_xs_decode"));
      for (int64_t ib32 = 0; ib32 < 8; ++ib32) {
        mlir::Value sc =
            loadIntAt(xb, inputPtrType, sizeLit(66 + ib32), constU8Type);
        mlir::Value db0 =
            fMul(fMul(d, fAdd(floatLit("0.5f"), i2f(iAnd(sc, intLit(0xf))))),
                 floatLit("0.25f"));
        mlir::Value db1 =
            fMul(fMul(d, fAdd(floatLit("0.5f"), i2f(iShr(sc, intLit(4))))),
                 floatLit("0.25f"));
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value q =
              u16ReadAt(xb, inputPtrType, sizeLit(2 + (4 * ib32 + l) * 2));
          mlir::Value gBase = iMul(u2i(uAnd(q, uLit(511))), intLit(8));
          mlir::Value sBase = iMul(u2i(uShr(q, uLit(9))), intLit(8));
          mlir::Value dbl = (l < 2) ? db0 : db1;
          int64_t outBase = ib32 * 32 + l * 8;
          elemLoop(8, [&](mlir::Value j) {
            mlir::Value ji = sz2i(j);
            mlir::Value g = tableByteAtInt(grid, iAdd(gBase, ji));
            mlir::Value sg = tableByteAtInt(signs, iAdd(sBase, ji));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(fMul(dbl, i2f(g)), i2f(sg)));
          });
        }
      }
    } else if (fmt == IQ2S) {
      // d@0, qs[64]@2, qh[8]@66, scales[8]@74; explicit sign region @34 (=qs+32).
      // grid-1024 (int64) + signs256 (indexed by the raw sign byte). Byte-exact to
      // dequantize_row_iq2_s.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value grid = castI8Base("tcrv_iq2s_grid", i64PtrType);
      mlir::Value signs = plainBase("tcrv_iq2s_signs256", i8PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq2_s_decode"));
      for (int64_t ib32 = 0; ib32 < 8; ++ib32) {
        mlir::Value sc =
            loadIntAt(xb, inputPtrType, sizeLit(74 + ib32), constU8Type);
        mlir::Value db0 =
            fMul(fMul(d, fAdd(floatLit("0.5f"), i2f(iAnd(sc, intLit(0xf))))),
                 floatLit("0.25f"));
        mlir::Value db1 =
            fMul(fMul(d, fAdd(floatLit("0.5f"), i2f(iShr(sc, intLit(4))))),
                 floatLit("0.25f"));
        mlir::Value qhv =
            loadIntAt(xb, inputPtrType, sizeLit(66 + ib32), constU8Type);
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value qsb = loadIntAt(xb, inputPtrType,
                                      sizeLit(2 + 4 * ib32 + l), constU8Type);
          // idx = qs[l] | ((qh[ib32] << (8-2l)) & 0x300).
          mlir::Value hb =
              iAnd(iShl(qhv, intLit(8 - 2 * l)), intLit(0x300));
          mlir::Value gBase = iMul(iOr(qsb, hb), intLit(8));
          mlir::Value sByte = loadIntAt(xb, inputPtrType,
                                        sizeLit(34 + 4 * ib32 + l), constU8Type);
          mlir::Value sBase = iMul(sByte, intLit(8));
          mlir::Value dbl = (l < 2) ? db0 : db1;
          int64_t outBase = ib32 * 32 + l * 8;
          elemLoop(8, [&](mlir::Value j) {
            mlir::Value ji = sz2i(j);
            mlir::Value g = tableByteAtInt(grid, iAdd(gBase, ji));
            mlir::Value sg = tableByteAtInt(signs, iAdd(sBase, ji));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(fMul(dbl, i2f(g)), i2f(sg)));
          });
        }
      }
    } else if (fmt == IQ3XXS) {
      // d@0, qs[96]@2, scales_and_signs @66 (=qs+64). grid-of-4 (uint32) + the
      // ksigns selector table. Byte-exact to dequantize_row_iq3_xxs.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value grid = castI8Base("tcrv_iq3xxs_grid", u32PtrType);
      mlir::Value ksigns = plainBase("tcrv_iq3xxs_ksigns", u8PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq3_xxs_decode"));
      for (int64_t ib32 = 0; ib32 < 8; ++ib32) {
        mlir::Value aux = u32ReadAt(xb, inputPtrType, sizeLit(66 + 4 * ib32));
        mlir::Value db = fMul(fMul(d, fAdd(floatLit("0.5f"),
                                           i2f(u2i(uShr(aux, uLit(28)))))),
                              floatLit("0.5f"));
        int64_t qsBase = 2 + 8 * ib32;
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value sel = u2i(uAnd(uShr(aux, uLit(7 * l)), uLit(127)));
          // signByte = ksigns[sel] (0..255).
          mlir::Value signByte =
              loadIntAt(ksigns, u8PtrType, i2sz(sel), constU8Type);
          mlir::Value g1Base = iMul(loadIntAt(xb, inputPtrType,
                                              sizeLit(qsBase + 2 * l + 0),
                                              constU8Type),
                                    intLit(4));
          mlir::Value g2Base = iMul(loadIntAt(xb, inputPtrType,
                                              sizeLit(qsBase + 2 * l + 1),
                                              constU8Type),
                                    intLit(4));
          int64_t outBase = ib32 * 32 + l * 8;
          elemLoop(4, [&](mlir::Value j) {
            mlir::Value ji = sz2i(j);
            // grid1[j] * sign(bit j); grid2[j] * sign(bit j+4).
            mlir::Value gv1 = tableByteAtInt(grid, iAdd(g1Base, ji));
            mlir::Value b0 = iAnd(iShr(signByte, ji), intLit(1));
            mlir::Value s0 = iSub(intLit(1), iMul(intLit(2), b0));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(fMul(db, i2f(gv1)), i2f(s0)));
            mlir::Value gv2 = tableByteAtInt(grid, iAdd(g2Base, ji));
            mlir::Value b1 =
                iAnd(iShr(signByte, iAdd(ji, intLit(4))), intLit(1));
            mlir::Value s1 = iSub(intLit(1), iMul(intLit(2), b1));
            storeF32At(yb, addSz(sizeLit(outBase + 4), j),
                       fMul(fMul(db, i2f(gv2)), i2f(s1)));
          });
        }
      }
    } else if (fmt == IQ3S) {
      // d@0, qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106. grid-of-4 (uint32,
      // 512) with a 9th index bit from qh; explicit sign bytes. Each outer group
      // g (0..3) runs TWO passes (qh[0] then qh[1]). Byte-exact to
      // dequantize_row_iq3_s.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value grid = castI8Base("tcrv_iq3s_grid", u32PtrType);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq3_s_decode"));
      auto pass = [&](mlir::Value qhByte, mlir::Value db, int64_t qsBase,
                      int64_t signsBase, int64_t outStart) {
        for (int64_t l = 0; l < 4; ++l) {
          mlir::Value q1 = loadIntAt(xb, inputPtrType,
                                     sizeLit(qsBase + 2 * l + 0), constU8Type);
          mlir::Value q2 = loadIntAt(xb, inputPtrType,
                                     sizeLit(qsBase + 2 * l + 1), constU8Type);
          // grid1 idx = q1 | ((qh << (8-2l)) & 256); grid2 idx uses (7-2l).
          mlir::Value g1Base = iMul(
              iOr(q1, iAnd(iShl(qhByte, intLit(8 - 2 * l)), intLit(256))),
              intLit(4));
          mlir::Value g2Base = iMul(
              iOr(q2, iAnd(iShl(qhByte, intLit(7 - 2 * l)), intLit(256))),
              intLit(4));
          mlir::Value signByte = loadIntAt(xb, inputPtrType,
                                           sizeLit(signsBase + l), constU8Type);
          int64_t outBase = outStart + l * 8;
          elemLoop(4, [&](mlir::Value j) {
            mlir::Value ji = sz2i(j);
            mlir::Value gv1 = tableByteAtInt(grid, iAdd(g1Base, ji));
            mlir::Value b0 = iAnd(iShr(signByte, ji), intLit(1));
            mlir::Value s0 = iSub(intLit(1), iMul(intLit(2), b0));
            storeF32At(yb, addSz(sizeLit(outBase), j),
                       fMul(fMul(db, i2f(gv1)), i2f(s0)));
            mlir::Value gv2 = tableByteAtInt(grid, iAdd(g2Base, ji));
            mlir::Value b1 =
                iAnd(iShr(signByte, iAdd(ji, intLit(4))), intLit(1));
            mlir::Value s1 = iSub(intLit(1), iMul(intLit(2), b1));
            storeF32At(yb, addSz(sizeLit(outBase + 4), j),
                       fMul(fMul(db, i2f(gv2)), i2f(s1)));
          });
        }
      };
      for (int64_t g = 0; g < 4; ++g) {
        mlir::Value scByte =
            loadIntAt(xb, inputPtrType, sizeLit(106 + g), constU8Type);
        mlir::Value db1 =
            fMul(d, i2f(iAdd(intLit(1),
                             iMul(intLit(2), iAnd(scByte, intLit(0xf))))));
        mlir::Value db2 =
            fMul(d, i2f(iAdd(intLit(1),
                             iMul(intLit(2), iShr(scByte, intLit(4))))));
        mlir::Value qh0 = loadIntAt(xb, inputPtrType, sizeLit(66 + 2 * g + 0),
                                    constU8Type);
        mlir::Value qh1 = loadIntAt(xb, inputPtrType, sizeLit(66 + 2 * g + 1),
                                    constU8Type);
        pass(qh0, db1, 2 + 16 * g, 74 + 8 * g, g * 64);
        pass(qh1, db2, 2 + 16 * g + 8, 74 + 8 * g + 4, g * 64 + 32);
      }
    }
  }

  return mlir::success();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
