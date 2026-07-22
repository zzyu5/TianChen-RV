#include "RVVToEmitCInternal.h"
#include "Weft/Conversion/RVV/RVVToEmitCSupport.h"
#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"
#include "Weft/Plugin/RVV/RVVFormulaDecision.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

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

// VariantToEmitCFunc forward-pass elementwise emit methods: ggml vec_scale /
// rmsnorm / silu (+ vexpf) / softmax / quantize_row_q8_0 / rope_norm. Split out
// of RVVToEmitC.cpp as a pure code move; the emitted C is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitTypedElementwiseLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The M-FLAT forward-elementwise scaffold's typed strip-loop lowering, the
    // constructed sibling of emitTypedFlatBlockDotLoopBody. The outer strip loop
    // is owned by the loop op; the per-strip map is re-emitted from the region's
    // core brick (anti-bypass: the brick's strip_index MUST be the loop induction
    // variable). BYTE-EXACT to the retired monolith weft_rvv.ggml_vec_scale_f32
    // emit modulo ONLY the source-op provenance token.
    weftrvv::TypedElementwiseLoopBodyOp loopBody;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto lb = llvm::dyn_cast<weftrvv::TypedElementwiseLoopBodyOp>(op))
        loopBody = lb;
    }
    if (!loopBody)
      return rewriter.notifyMatchFailure(
          scope, "typed elementwise loop body missing the op");

    // The "reduce" model carries a loop-carried accumulator: the FIRST forward
    // REDUCE operator constructed through the scaffold is rms_norm (the Σx²
    // scalar-double fold + the scalar rsqrt + the vectorized normalize strip),
    // whose per-element fold rides the weft_rvv.elementwise_rms_norm_reduce_core
    // reduce-core brick. Dispatch to its dedicated re-emit before the map path.
    if (loopBody.getReduceMapModel() == "reduce") {
      weftrvv::ElementwiseRmsNormReduceCoreOp rmsCore;
      loopBody.getBody().walk([&](weftrvv::ElementwiseRmsNormReduceCoreOp o) {
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
    // the weft_rvv.elementwise_rope_rotate_core rotate-core brick. Dispatch to its
    // dedicated re-emit before the map path.
    if (loopBody.getReduceMapModel() == "rotate") {
      weftrvv::ElementwiseRopeRotateCoreOp ropeCore;
      loopBody.getBody().walk([&](weftrvv::ElementwiseRopeRotateCoreOp o) {
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
    // (elementwise_scale_map, y[i] *= v, in-place single buffer), the silu map
    // (elementwise_silu_map, y[i] = x[i]*sigmoid(x[i]), a two-buffer x->y map), the
    // BINARY map (elementwise_binary_map, z[i] = x[i]{+|*}y[i], the add/mul support
    // ops), the COPY map (elementwise_copy_map, y[i] = x[i], the cpy support op), or
    // the GELU map (elementwise_gelu_map, y[i] = gelu(x[i]), the gelu support op).
    // Find the brick + yield.
    weftrvv::ElementwiseScaleMapOp mapOp;
    weftrvv::ElementwiseSiluMapOp siluOp;
    weftrvv::ElementwiseBinaryMapOp binaryOp;
    weftrvv::ElementwiseCopyMapOp copyOp;
    weftrvv::ElementwiseGeluMapOp geluOp;
    weftrvv::TypedElementwiseLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::ElementwiseScaleMapOp>(bodyOp))
        mapOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::ElementwiseSiluMapOp>(bodyOp))
        siluOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::ElementwiseBinaryMapOp>(bodyOp))
        binaryOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::ElementwiseCopyMapOp>(bodyOp))
        copyOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::ElementwiseGeluMapOp>(bodyOp))
        geluOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedElementwiseLoopYieldOp>(bodyOp))
        yieldOp = o;
    });
    if (!yieldOp || (!mapOp && !siluOp && !binaryOp && !copyOp && !geluOp))
      return rewriter.notifyMatchFailure(
          loopBody, "map-model elementwise loop body requires a recognized map "
                    "core brick (elementwise_scale_map | elementwise_silu_map | "
                    "elementwise_binary_map | elementwise_copy_map | "
                    "elementwise_gelu_map) + the loop yield");

    // The SILU map reuses the SAME outer strip-loop op + map model, but its
    // per-strip decode is the m2 exp polynomial over two buffers (x->y), so it
    // owns a dedicated core-brick emit. Dispatch to it before the scale path.
    if (siluOp)
      return emitElementwiseSiluMapStrip(rewriter, loc, loopBody, siluOp, avlArg,
                                         sizeType, valueMap);

    // The forward SUPPORT-op maps (add/mul binary, cpy copy, gelu scalar) each own
    // a dedicated re-emit that sources the ABI from the region brick (anti-bypass)
    // and delegates to the SHARED byte-exact strip/loop body.
    if (binaryOp)
      return emitElementwiseBinaryMapStrip(rewriter, loc, loopBody, binaryOp,
                                           avlArg, sizeType, valueMap);
    if (copyOp)
      return emitElementwiseCopyMapStrip(rewriter, loc, loopBody, copyOp, avlArg,
                                         sizeType, valueMap);
    if (geluOp)
      return emitElementwiseGeluMapStrip(rewriter, loc, loopBody, geluOp, avlArg,
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

    llvm::StringRef opName = mapOp.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = mapOp.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type bufferPtrType = buffer.getType();

    // The f32 strip-loop LMUL is a bounded final construction fact. It is the
    // *how* (vector grouping /
    // strip width), never the *what*: the result is byte-exact at any anchor
    // (every lane is multiplied by the same scalar v). The verifier bounds it to
    // m1|m2|m4|m8 (carried as "strip_lmul" so the op holds no forbidden
    // dataflow-parameter "lmul" at the I5 boundary).
    if (!mapOp.getStripLmul())
      return rewriter.notifyMatchFailure(
          mapOp, "elementwise_scale_map reached emission without final "
                 "strip_lmul");
    llvm::StringRef lmul = *mapOp.getStripLmul();
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
    weftrvv::TypedElementwiseLoopBodyOp loopBody,
    weftrvv::ElementwiseRmsNormReduceCoreOp rmsCore, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The CONSTRUCTED rms_norm reduce-model body, the reduce sibling of the map
    // paths (scale/silu) and of the block-dot loop scaffold. The outer loop op
    // owns the reduce shape (reduce_map_model "reduce": a loop-carried f64
    // accumulator region arg + the yield that carries it back); this re-emit
    // sources the WHOLE rms_norm ABI + the byte-exact scalar-double Σx² fold /
    // scalar rsqrt / vectorized normalize strip from the region's reduce core
    // brick (anti-bypass). BYTE-EXACT to the retired monolith
    // weft_rvv.ggml_rms_norm_f32 emit modulo ONLY the source-op provenance token.

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

    llvm::StringRef opName = rmsCore.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = rmsCore.getWEFTEmitCLowerableSourceRole();
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
    weftrvv::ElementwiseMulMapOp fusedMul;
    rmsCore.getEpilogue().walk(
        [&](weftrvv::ElementwiseMulMapOp o) { fusedMul = o; });
    weftrvv::ElementwiseQuantizeQ80MapOp quantBrick;
    if (fusedMul)
      fusedMul.getQuantEpilogue().walk(
          [&](weftrvv::ElementwiseQuantizeQ80MapOp o) { quantBrick = o; });

    if (quantBrick) {
      mlir::Value weight = valueMap.lookup(fusedMul.getWeight());
      mlir::Value yq8 = valueMap.lookup(quantBrick.getOutput());
      if (!weight || !yq8)
        return rewriter.notifyMatchFailure(
            quantBrick, "fused quant epilogue ABI operand unmapped");

      llvm::StringRef mulOpName = fusedMul.getWEFTEmitCLowerableSourceOpName();
      llvm::StringRef mulRole = fusedMul.getWEFTEmitCLowerableSourceRole();
      llvm::StringRef qOpName = quantBrick.getWEFTEmitCLowerableSourceOpName();
      llvm::StringRef qRole = quantBrick.getWEFTEmitCLowerableSourceRole();

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
    // strip LMUL is a bounded final construction fact. It is byte-exact at any anchor (a
    // bare per-lane vfmul_vf -- no FMA, no reduction). This is the SAME strip
    // machinery F1 (scale) emits, except two-buffer (x in, y out) instead of
    // in-place: byte-identical (both one f32 multiply per lane), avoiding ggml's
    // memcpy+in-place-scale.
    if (!rmsCore.getStripLmul())
      return rewriter.notifyMatchFailure(
          rmsCore, "rms_norm core reached emission without final strip_lmul");
    llvm::StringRef lmul = *rmsCore.getStripLmul();
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
      // carries a weft_rvv.elementwise_mul_map brick in its $epilogue region, the
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
      weftrvv::ElementwiseMulMapOp mulMap;
      rmsCore.getEpilogue().walk(
          [&](weftrvv::ElementwiseMulMapOp o) { mulMap = o; });

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
        llvm::StringRef mulOpName = mulMap.getWEFTEmitCLowerableSourceOpName();
        llvm::StringRef mulRole = mulMap.getWEFTEmitCLowerableSourceRole();
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

    // Fast-path result (all lanes |n|<=126): return k + j*k directly. ggml seeds
    // its early return with EXACTLY this (`vfmacc_vv(k, j, k)`); we reuse it both
    // as the vcpop==0 result AND as the slow-path r1 c-false lane, so the emit is
    // byte-identical to the retired unconditional slow path (whose c-false /
    // |n|<=192 lanes were already bitwise-equal to k + j*k). vec.h:1348.
    mlir::Value r1False = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                               mlir::ValueRange{k, k, j, bodyVL});

    // RESTORED ggml short-circuit (vec.h:1348 `if (!vcpop(c)) return fast`): emit
    // the ~14-op slow-path merge INSIDE a data-dependent `emitc.if`, guarded by
    // `__riscv_vcpop_m_b16(c, vl) != 0`. On the common all-fast strip (every soft_max
    // / silu decode input) the guard is 0 and the slow path is SKIPPED -- the
    // scheduling maturity that closes the parity-by-adoption census gap. BYTE-EXACT:
    // the result variable is seeded with the fast path, so vcpop==0 yields exactly
    // the value the retired unconditional emit produced (see above). Reuses the
    // STRUCTURED emitc.variable + emitc.if idiom the q8_0/K-quant `id`/`amax`
    // conditionals use (NOT a raw string).
    mlir::Type i1Type = rewriter.getI1Type();
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("expf_r", opName, role));
    auto resultVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32VecType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(loc, resultVar, r1False);

    // size_t pop = __riscv_vcpop_m_b16(c, vl);
    mlir::Value pop =
        vcall(sizeType, "__riscv_vcpop_m_b16", mlir::ValueRange{c, bodyVL});
    mlir::Value zeroPop =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value anyExtreme = rewriter.create<emitc::CmpOp>(
        loc, i1Type, emitc::CmpPredicate::ne, pop, zeroPop);
    auto slowIf = rewriter.create<emitc::IfOp>(loc, anyExtreme,
                                               /*addThenBlock=*/true,
                                               /*addElseBlock=*/false);
    {
      mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
      rewriter.setInsertionPointToStart(&slowIf.getThenRegion().front());
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
      //     vfmacc_vv(k, k, j, vl),   <- r1False (fast, computed above)
      //     vfmul_vv(vfmacc_vv(s2, s2, j, vl), s1, vl),
      //     c, vl);
      mlir::Value r1TrueInner = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                                      mlir::ValueRange{s2, s2, j, bodyVL});
      mlir::Value r1True = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                                 mlir::ValueRange{r1TrueInner, s1, bodyVL});
      mlir::Value r1 = vcall(f32VecType, "__riscv_vmerge_vvm_f32m2",
                             mlir::ValueRange{r1False, r1True, c, bodyVL});
      // result = vmerge_vvm(r1, vfmul_vv(s1, s1, vl),
      //                     vmfgt_vf(vfabs_v(n), 192.0f, vl), vl);
      mlir::Value absN2 = vcall(f32VecType, "__riscv_vfabs_v_f32m2",
                                mlir::ValueRange{n, bodyVL});
      mlir::Value overMask = vcall(boolType, "__riscv_vmfgt_vf_f32m2_b16",
                                   mlir::ValueRange{absN2, fimm("192.0f"),
                                                    bodyVL});
      mlir::Value s1Sq = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                               mlir::ValueRange{s1, s1, bodyVL});
      mlir::Value slow = vcall(f32VecType, "__riscv_vmerge_vvm_f32m2",
                               mlir::ValueRange{r1, s1Sq, overMask, bodyVL});
      rewriter.create<emitc::VerbatimOp>(loc, assignComment("expf_r", opName, role));
      rewriter.create<emitc::AssignOp>(loc, resultVar, slow);
      rewriter.create<emitc::YieldOp>(loc);
    }
    return rewriter.create<emitc::LoadOp>(loc, f32VecType, resultVar)
        .getResult();
  }

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseSiluMapStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::TypedElementwiseLoopBodyOp loopBody,
    weftrvv::ElementwiseSiluMapOp siluOp, mlir::Value avlArg,
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

    llvm::StringRef opName = siluOp.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = siluOp.getWEFTEmitCLowerableSourceRole();
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
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The CONSTRUCTED soft_max reduce-model body, the exp-sum-reduce sibling of
    // the rms_norm Σx² reduce (emitElementwiseRmsNormReduceStrip) and of the
    // scale/silu map paths. The outer loop op owns the reduce shape
    // (reduce_map_model "reduce": a loop-carried f64m1 WIDENING accumulator region
    // arg + the yield that carries it back); this re-emit sources the WHOLE
    // soft_max ABI + the byte-exact fused exp-store-widening-reduce strip from the
    // region's reduce core brick (anti-bypass). BYTE-EXACT to the retired monolith
    // weft_rvv.ggml_vec_soft_max_f32 emit modulo ONLY the source-op provenance
    // token. Returns the f64 sum value (the dispatch wraps it in the function's
    // `return`).
    weftrvv::TypedElementwiseLoopBodyOp loopBody;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto lb = llvm::dyn_cast<weftrvv::TypedElementwiseLoopBodyOp>(op))
        loopBody = lb;
    }
    if (!loopBody)
      return rewriter.notifyMatchFailure(
          scope, "soft_max reduce body missing the typed elementwise loop op");
    weftrvv::ElementwiseSoftMaxReduceCoreOp softMaxCore;
    loopBody.getBody().walk(
        [&](weftrvv::ElementwiseSoftMaxReduceCoreOp o) { softMaxCore = o; });
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

    llvm::StringRef opName = softMaxCore.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = softMaxCore.getWEFTEmitCLowerableSourceRole();
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

// The SHARED q8_0 quantize_row block-encode body: the AoS `nb = n/32` block loop,
// the per-block f32 load in ONE e32m8 strip, and the amax/scale/narrow q8_0 core
// (emitQuantizeQ80BlockBody). Extracted VERBATIM from the block-loop tail of the
// retired direct q8_0 monolith so the CONSTRUCTED typed lowering (via
// emitTypedQuantizeRowLoopBody) emits byte-identical C (modulo only the source-op
// provenance token threaded through opName/role). Byte-exact to ggml's EXACT RVV
// method (riscv/quants.c:32-71). Streaming (no cross-block accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitQuantizeRowQ80BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, int64_t qk, int64_t blockStride,
    int64_t scaleOffset, int64_t quantOffset, llvm::StringRef opName,
    llvm::StringRef role) const {

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    // The block's 32 lanes ride in ONE e32m8 strip (ggml's `size_t vl = QK8_0`).
    // The wide-f32 load type is ggml's exact path type; the per-block
    // amax/scale/narrow body's remaining types live in emitQuantizeQ80BlockBody.
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
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

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
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

// The SHARED q8_1 quantize_row block-encode body: the q8_0 amax/scale/narrow SIBLING
// PLUS the extra vwredsum integer block sum stored as the fp16 block_q8_1.s.
// Extracted VERBATIM from the block-loop tail of the retired direct q8_1 monolith
// monolith so the CONSTRUCTED typed lowering (via emitTypedQuantizeRowLoopBody) emits
// byte-identical C (modulo only the source-op provenance token threaded through
// opName/role). Byte-exact to ggml's EXACT RVV method. Streaming (no cross-block
// accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitQuantizeRowQ81BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, int64_t qk, int64_t blockStride,
    int64_t scaleOffset, int64_t quantOffset, llvm::StringRef opName,
    llvm::StringRef role) const {
    // q8_1's extra fp16 block-sum slot is intrinsic to this typed leaf.
    const int64_t sumOffset = 2;

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

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
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

// The SHARED q8_K quantize_row super-block-encode body: the QK_K=256 min/max-
// symmetric quantizer -- the vsetvlmax_e32m8-folded min/max strip loop + scalar
// reduce, the fabsf-symmetric iscale, the STRUCTURED amax==0 zero path (float d=0,
// memset qs+bsums), else the FLOAT d store + the quantize strip (vfmul + vfcvt/vnclip
// RNE narrow) + the 256 int8 qs store + the 16 per-16 vslidedown-advanced vwredsum
// bsums. Extracted VERBATIM from the loop tail of the retired direct q8_K monolith
// monolith so the CONSTRUCTED typed lowering (via emitTypedQuantizeRowLoopBody) emits
// byte-identical C (modulo only the source-op provenance token threaded through
// opName/role). Byte-exact to ggml's EXACT RVV method. Streaming (no cross-block
// accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitQuantizeRowQ8KBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, int64_t qk, int64_t blockStride,
    int64_t scaleOffset, int64_t quantOffset, llvm::StringRef opName,
    llvm::StringRef role) const {
    // q8_K's per-16 bsums tail is intrinsic to this typed leaf.
    const int64_t bsumsOffset = 260;

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

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
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

// Lower the CONSTRUCTED streaming quantize_row region: walk the
// weft_rvv.typed_quantize_row_loop_body, extract its per-block ENCODE brick
// (weft_rvv.quantize_row_encode_core) + the VOID yield, enforce the anti-bypass
// invariant (the brick's block_index MUST be the region induction variable / region
// arg 0, so the ABI bases are sourced from the BRICK not inferred), and re-emit the
// whole nb block loop + per-block encode via the SHARED body emitter -- byte-exact to
// the retired dispatch-wired per-format quantize monolith. The MIRROR of
// emitTypedDequantizeRowLoopBody (f32->QUANT rather than QUANT->f32).
mlir::LogicalResult VariantToEmitCFunc::emitTypedQuantizeRowLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  weftrvv::TypedQuantizeRowLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<weftrvv::TypedQuantizeRowLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed quantize_row loop body missing the op");

  weftrvv::QuantizeRowEncodeCoreOp coreOp;
  weftrvv::TypedQuantizeRowLoopYieldOp yieldOp;
  loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
    if (auto o = llvm::dyn_cast<weftrvv::QuantizeRowEncodeCoreOp>(bodyOp))
      coreOp = o;
    else if (auto o =
                 llvm::dyn_cast<weftrvv::TypedQuantizeRowLoopYieldOp>(bodyOp))
      yieldOp = o;
  });
  mlir::Block &coreBlock = loopBody.getBody().front();
  if (!coreOp || !yieldOp)
    return rewriter.notifyMatchFailure(
        loopBody, "typed quantize_row body requires the "
                  "quantize_row_encode_core brick + the void loop yield");
  if (coreBlock.getNumArguments() != 1)
    return rewriter.notifyMatchFailure(
        loopBody, "typed quantize_row body region must carry exactly the "
                  "block_index induction variable");
  mlir::Value blockIndex = coreBlock.getArgument(0);
  if (coreOp.getBlockIndex() != blockIndex)
    return rewriter.notifyMatchFailure(
        loopBody, "the encode-core brick's block_index must be the loop "
                  "induction variable (region arg 0) so the emit addresses "
                  "base + ib*stride, not block-0");

  // Anti-bypass (I7): the ABI bases are sourced from the BRICK's operands.
  mlir::Value input = valueMap.lookup(coreOp.getInput());
  mlir::Value output = valueMap.lookup(coreOp.getOutput());
  if (!input || !output)
    return rewriter.notifyMatchFailure(loopBody,
                                       "typed quantize_row ABI operand unmapped");

  llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
  const int64_t qk = coreOp.getQkAttr().getInt();
  const int64_t blockStride = coreOp.getBlockStrideAttr().getInt();
  const int64_t scaleOffset = coreOp.getScaleByteOffsetAttr().getInt();
  const int64_t quantOffset = coreOp.getQuantByteOffsetAttr().getInt();

  // Mechanical dispatch on the closed typed formula result. Construction
  // provenance is deliberately not read here.
  switch (coreOp.getQuantizeLeaf()) {
  case weftrvv::QuantizeRowLeaf::Q8_0:
    return emitQuantizeRowQ80BodyShared(
        rewriter, loc, input, output, avlArg, sizeType, qk, blockStride,
        scaleOffset, quantOffset, opName, role);
  case weftrvv::QuantizeRowLeaf::Q8_1:
    return emitQuantizeRowQ81BodyShared(rewriter, loc, input, output, avlArg,
                                        sizeType, qk, blockStride, scaleOffset,
                                        quantOffset, opName, role);
  case weftrvv::QuantizeRowLeaf::Q8_K:
    return emitQuantizeRowQ8KBodyShared(rewriter, loc, input, output, avlArg,
                                        sizeType, qk, blockStride, scaleOffset,
                                        quantOffset, opName, role);
  }
  return rewriter.notifyMatchFailure(loopBody,
                                     "unknown typed quantize_row leaf");
}

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseRopeRotateStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::TypedElementwiseLoopBodyOp loopBody,
    weftrvv::ElementwiseRopeRotateCoreOp ropeCore, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The CONSTRUCTED rope rotate-model body, the per-pair-recurrence sibling of
    // the map paths (scale/silu) and the reduce paths (rms_norm/soft_max). The
    // outer loop op owns the rotate shape (reduce_map_model "rotate": a per-pair
    // scalar loop with a loop-carried f32 theta region arg + the yield that
    // carries it back); this re-emit sources the WHOLE rope ABI + the byte-exact
    // scalar cos/sin angle seam / position-dependent 2x2 rotation / f32 theta
    // recurrence from the region's rotate core brick (anti-bypass). BYTE-EXACT to
    // the retired monolith weft_rvv.ggml_rope_norm_f32 emit modulo ONLY the
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

    llvm::StringRef opName = ropeCore.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = ropeCore.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type indexType = rewriter.getIndexType();

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

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

// The ONE byte-exact m8 per-lane MAP strip, shared by the CONSTRUCTED typed-region
// binary/copy re-emits (emitElementwise{Binary,Copy}MapStrip). Byte-exact to the
// retired support-op monolith emit modulo the source-op provenance token. See the
// header for the WHY.
mlir::LogicalResult VariantToEmitCFunc::emitForwardVecMapStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::ValueRange inputs, mlir::Value output, llvm::StringRef opName,
    llvm::StringRef role, llvm::StringRef binaryCallee, mlir::Type sizeType,
    mlir::Value avlArg) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type constFloatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
  mlir::Type floatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
  // The bare per-lane add/mul/cpy are byte-exact at any LMUL (no reduction), so m8
  // is a fixed resource fact ggml's apply path uses, not a knob.
  llvm::StringRef lmul = "m8";
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
  std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
  std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
  std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");

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
}

// The ONE byte-exact SCALAR per-element tanh gelu loop, shared by the support emit
// AND the constructed gelu re-emit. tanhf is the sanctioned scalar-libm opaque seam
// (the sibling of rope's cosf/sinf, rms_norm's sqrtf).
mlir::LogicalResult VariantToEmitCFunc::emitForwardGeluScalarLoop(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, llvm::StringRef opName,
    llvm::StringRef role, mlir::Type sizeType, mlir::Value avlArg,
    bool f16Lut) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type constFloatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
  mlir::Type floatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
  mlir::Type indexType = rewriter.getIndexType();
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

    // The per-element gelu value `gv`. Two precision tiers:
    //  - default: the ggml reference EXACT tanh gelu (ggml_gelu_f32),
    //      0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))
    //  - f16Lut (G.0.3 same-precision-tier rematch): the SAME numeric contract as
    //      ggml's as-shipped GGML_GELU_FP16 f16 lookup table, via the
    //      `weft_gelu_f16lut_scalar` opaque seam (module preamble). The table is
    //      pure memoization of that seam, so the tableless call is bit-identical to
    //      the shipped LUT opponent (byte-exact same-tier oracle).
    mlir::Value gv;
    if (f16Lut) {
      gv = emitOpaqueCall(rewriter, loc, floatType, "weft_gelu_f16lut_scalar",
                          mlir::ValueRange{xv}, opName, role);
    } else {
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
      gv = rewriter.create<emitc::MulOp>(loc, floatType, halfX, onePlusTanh);
    }

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

// NOTE: emitGgmlForwardElementwiseF32 (the DISPATCH-WIRED support-op monolith
// dispatcher over weft_rvv.{vec_add,vec_mul,vec_cpy,gelu}_f32) was RETIRED at the
// support flip (dispatch-wired -> constructed, C_construct 73->77). add/mul/cpy/gelu
// are now CONSTRUCTED through the abstract weft_rvv.ggml_forward_elementwise source
// op + the pre-emitc front door, re-emitted below from the region core brick by
// emitElementwise{Binary,Copy,Gelu}MapStrip -- which delegate to the SAME SHARED
// byte-exact helpers (emitForwardVecMapStrip / emitForwardGeluScalarLoop) the retired
// dispatcher called, so the emitted C is byte-identical modulo ONLY the source-op
// provenance token.

// The CONSTRUCTED forward BINARY / COPY / GELU re-emits: source the ABI from the
// region's core brick (anti-bypass: the strip_index MUST be the loop induction
// variable), then delegate to the SHARED byte-exact strip/loop body. Byte-exact to
// the support-op emit modulo the source-op provenance token
// (weft_rvv.{vec_add,vec_mul,vec_cpy,gelu}_f32 ->
// weft_rvv.elementwise_{binary,copy,gelu}_map).
mlir::LogicalResult VariantToEmitCFunc::emitElementwiseBinaryMapStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::TypedElementwiseLoopBodyOp loopBody,
    weftrvv::ElementwiseBinaryMapOp binaryOp, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::Value stripIndex = loopBody.getBody().front().getArgument(0);
  if (binaryOp.getStripIndex() != stripIndex)
    return rewriter.notifyMatchFailure(
        binaryOp, "the elementwise_binary_map brick's strip_index must be the "
                  "loop induction variable (region arg 0)");
  mlir::Value lhs = valueMap.lookup(binaryOp.getLhs());
  mlir::Value rhs = valueMap.lookup(binaryOp.getRhs());
  mlir::Value output = valueMap.lookup(binaryOp.getOutput());
  if (!lhs || !rhs || !output)
    return rewriter.notifyMatchFailure(binaryOp,
                                       "binary-map ABI operand unmapped");
  // binary_op "add" -> vfadd_vv, "mul" -> vfmul_vv (the verifier bounds it).
  llvm::StringRef intrin = binaryOp.getBinaryOp() == "add" ? "vfadd" : "vfmul";
  return emitForwardVecMapStrip(
      rewriter, loc, {lhs, rhs}, output,
      binaryOp.getWEFTEmitCLowerableSourceOpName(),
      binaryOp.getWEFTEmitCLowerableSourceRole(),
      riscvIntrinsicName(intrin, 32, "m8", "f32"), sizeType, avlArg);
}

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseCopyMapStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::TypedElementwiseLoopBodyOp loopBody,
    weftrvv::ElementwiseCopyMapOp copyOp, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::Value stripIndex = loopBody.getBody().front().getArgument(0);
  if (copyOp.getStripIndex() != stripIndex)
    return rewriter.notifyMatchFailure(
        copyOp, "the elementwise_copy_map brick's strip_index must be the loop "
                "induction variable (region arg 0)");
  mlir::Value input = valueMap.lookup(copyOp.getInput());
  mlir::Value output = valueMap.lookup(copyOp.getOutput());
  if (!input || !output)
    return rewriter.notifyMatchFailure(copyOp, "copy-map ABI operand unmapped");
  return emitForwardVecMapStrip(rewriter, loc, {input}, output,
                                copyOp.getWEFTEmitCLowerableSourceOpName(),
                                copyOp.getWEFTEmitCLowerableSourceRole(),
                                /*binaryCallee=*/"", sizeType, avlArg);
}

mlir::LogicalResult VariantToEmitCFunc::emitElementwiseGeluMapStrip(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::TypedElementwiseLoopBodyOp loopBody,
    weftrvv::ElementwiseGeluMapOp geluOp, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::Value stripIndex = loopBody.getBody().front().getArgument(0);
  if (geluOp.getStripIndex() != stripIndex)
    return rewriter.notifyMatchFailure(
        geluOp, "the elementwise_gelu_map brick's strip_index must be the loop "
                "induction variable (region arg 0)");
  mlir::Value input = valueMap.lookup(geluOp.getInput());
  mlir::Value output = valueMap.lookup(geluOp.getOutput());
  if (!input || !output)
    return rewriter.notifyMatchFailure(geluOp, "gelu-map ABI operand unmapped");
  // G.0.3 same-precision-tier variant: the optional discardable attr
  // `gelu_precision = "f16lut"` selects the f16-LUT numeric tier (byte-exact to the
  // as-shipped GGML_GELU_FP16 opponent) instead of the default exact-tanhf tier.
  bool f16Lut = false;
  if (auto prec = geluOp->getAttrOfType<mlir::StringAttr>("gelu_precision"))
    f16Lut = prec.getValue() == "f16lut";
  return emitForwardGeluScalarLoop(
      rewriter, loc, input, output,
      geluOp.getWEFTEmitCLowerableSourceOpName(),
      geluOp.getWEFTEmitCLowerableSourceRole(), sizeType, avlArg, f16Lut);
}

// The OWNED REAL-VECTOR 4-bit nibble dequantize_row block-decode body for the flat
// legacy formats (q4_0/q5_0 the single-mul SAFE set, q4_1/q5_1 the min-add FMA set).
// The ISSUE-001 / [L-8] reverse of the scalar emitDequantizeRowNibbleBodyShared (whose
// owned vector-intrinsic count is 0 -- the host-autovec codegen-lottery owns any
// vectorization there): here the vector content is the EMITTER'S (OWNED __riscv_v
// intrinsics), de-lottery by construction. Per block, the 16 packed nibble bytes decode
// as TWO 16-lane half-block pipelines (low nibbles `qs[j]&0x0F` -> y[j], high nibbles
// `qs[j]>>4` -> y[j+16]): vle8 (the 16 packed bytes, u8m1) -> vand_vx/vsrl_vx (the two
// nibble planes, u8m1) -> vzext_vf4 (nibble 0..15 -> u32m4) -> [q5 5th-bit spread:
// vid/vmv/vsrl_vv/vand_vx/vsll_vx/vor_vv, bit j (resp j+16) of the byte-assembled uint32
// qh -> {0,16}] -> vreinterpret to i32m4 -> [q4_0/q5_0 pre-scale bias vsub_vx: -8/-16] ->
// vfcvt_f_x_v (i32->f32, exact for |val|<=31) -> the runtime `d` scale (vfmul_vf for the
// single-mul set) OR the fused min-add (vfmv_v_f(m) + vfmacc_vf(d): `d*val + m` in ONE
// rounding, matching the CONTRACTED scalar `x0*d+m` the -ffp-contract=on opponent
// autovec's to vfmadd) -> vse32 (the 16-float contiguous half-block store). NO gather
// (nibble unpack is not a codebook lookup) -> NO HW-gather wall (contrast the iq3_xxs
// grid leaf). The 16-lane half-block width is the FIXED QK/2 nibble geometry (NOT a
// tunable knob); the pipeline LMULs (u8m1 for the 16 packed bytes, u32m4/i32m4/f32m4 for
// the 4x-widened 16-lane int->float pipeline) are DERIVED from that width, NOT literal
// knobs. Byte-exact-vs-ggml-reference dequantize_row_<fmt> by construction: the fp16
// d (+ optional m) seam is the SAME `(float)*(const _Float16 *)` read, the nibble/5th-bit
// integer values are byte-identical to the scalar `int` decode (all in [-16,31], exact in
// f32), and the fold rounds identically (single-mul for q4_0/q5_0 -> no fp-contraction
// ambiguity; single fused mul-add for q4_1/q5_1 -> matches the contracted opponent).
// Only the CONSTRUCTED path (emitTypedDequantizeRowLoopBody) routes here; the
// dispatch-wired monolith fallback keeps the scalar shared body (the q8_0/iq3_xxs
// precedent). Streaming sibling of emitDequantizeRowQ8_0VectorBody (no accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowNibbleVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::NibbleDecodePlan &plan) const {
  // The decode tuple + strip geometry are read from the selected NibbleDecode plan
  // (produced by decideNibbleDecode) instead of retired per-format scalar params.
  const int64_t stride = plan.weightBlockStride;
  const int64_t dOff = plan.scaleByteOffset;
  const int64_t mOff = plan.minByteOffset;
  const int64_t qhOff = plan.qhByteOffset;
  const int64_t qsOff = plan.quantByteOffset;
  const int64_t sub = plan.nibbleBias;
  const bool hasMin = plan.hasMin;
  const bool hasQh = plan.hasQh;
  // block_q4_0/q4_1/q5_0/q5_1 AoS facts: qk=32 lanes per block, 16 packed nibble bytes
  // (the byte-exact ggml ABI shape constants, NOT tunable knobs -- the SAME facts the
  // scalar emitDequantizeRowNibbleBodyShared hard-codes). The half-block strip lane
  // count rides the plan (plan.stripLanes == qk/2 == 16, analytic g), so a plan
  // change to stripLanes changes the emitted per-strip vl.
  const int64_t qk = 32, half = qk / 2;
  const int64_t stripLanes = plan.stripLanes;

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type floatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
  // The 16-lane half-block pipeline types: u8m1 holds the 16 packed nibble bytes
  // (VLEN>=128: m1 SEW8 >= 16 lanes); the vf4 4x widening lands the 0..15 nibble in
  // u32m4 (VLEN128 m4 SEW32 == 16 lanes exactly), then i32m4 (reinterpret) for the bias
  // + f32m4 for the convert/scale/store. All DERIVED from the fixed QK/2=16 nibble width.
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
  mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m4_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m4_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  // vx / v_x scalar operands: a plain integer literal of the matching opaque C type
  // (call_opaque emits it verbatim; C converts to the intrinsic's rhs type).
  mlir::Type u8ScalarType = emitc::OpaqueType::get(ctx, "uint8_t");
  mlir::Type u32ScalarType = emitc::OpaqueType::get(ctx, "uint32_t");
  mlir::Type i32ScalarType = emitc::OpaqueType::get(ctx, "int32_t");
  auto u8Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, u8ScalarType, std::to_string(v));
  };
  auto u32Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, u32ScalarType, std::to_string(v));
  };
  auto i32Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, i32ScalarType, std::to_string(v));
  };
  auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) {
    return emitLoadByteAsUint(rewriter, loc, constU8Type, uintType, ptr, i);
  };
  // The per-strip vl rides the plan (plan.stripLanes, reproduce-current == half == 16):
  // the load-bearing witness that the emit consumes the PLAN's DERIVED geometry.
  mlir::Value halfVl = sizeLit(stripLanes);

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // size_t nb = k / 32;  (ggml's `const int nb = k / qk`; k % qk == 0, no tail).
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1),
                                                /*bodyBuilder=*/nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    // const uint8_t *xb = x + ib*stride;  float *yb = (float *)(y + ib*32);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "x_block"));
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

    // float d = (float)*(const _Float16 *)(xb + dOff);  (the fp16 block scale).
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
    mlir::Value dAddr = xb;
    if (dOff != 0)
      dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(dOff));
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{dAddr}, opName, role,
                                   llvm::StringRef("fcvt.s.h"));

    // float m = (float)*(const _Float16 *)(xb + mOff);  (q4_1/q5_1 block min).
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
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "qh"));
      mlir::Value qhBaseRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qhOff));
      mlir::Value qhBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qhBaseRaw).getResult();
      mlir::Value b0 = loadByteAsUint(qhBase, 0);
      mlir::Value b1 = loadByteAsUint(qhBase, 1);
      mlir::Value b2 = loadByteAsUint(qhBase, 2);
      mlir::Value b3 = loadByteAsUint(qhBase, 3);
      mlir::Value s1 = rewriter.create<emitc::BitwiseLeftShiftOp>(
          loc, uintType, b1, u32Lit(8));
      mlir::Value s2 = rewriter.create<emitc::BitwiseLeftShiftOp>(
          loc, uintType, b2, u32Lit(16));
      mlir::Value s3 = rewriter.create<emitc::BitwiseLeftShiftOp>(
          loc, uintType, b3, u32Lit(24));
      qh = rewriter.create<emitc::BitwiseOrOp>(loc, uintType, b0, s1).getResult();
      qh = rewriter.create<emitc::BitwiseOrOp>(loc, uintType, qh, s2).getResult();
      qh = rewriter.create<emitc::BitwiseOrOp>(loc, uintType, qh, s3).getResult();
    }

    // const uint8_t *qs = (const uint8_t *)(xb + qsOff);  (the 16 packed nibbles).
    mlir::Value qsBaseRaw =
        rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff));
    mlir::Value qsBase =
        rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBaseRaw).getResult();

    // vuint8m1_t qv = vle8_v_u8m1(qs, 16);  (the 16 packed nibble bytes.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "nibble_load"));
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, "m1", "u8");
    mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, u8LoadCallee,
                                    mlir::ValueRange{qsBase, halfVl}, opName, role);

    // The half-block pipeline: nibble plane (u8m1) -> u32m4 (zext) -> [q5 5th-bit
    // merge] -> i32m4 (reinterpret) -> [bias sub] -> f32m4 (convert) -> fold -> store.
    // `qhShiftBase` = 0 for the low half (bit j of qh), 16 for the high half (bit
    // j+16); `yStore` is the block output cursor (yb for the low nibbles, yb+16 for
    // the high nibbles).
    auto emitHalf = [&](mlir::Value nibblePlane, int64_t qhShiftBase,
                        mlir::Value yStore) {
      // n_u32 = vzext_vf4(nibblePlane);  (0..15, exact zero-extend).
      mlir::Value n_u32 = emitOpaqueCall(
          rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
          mlir::ValueRange{nibblePlane, halfVl}, opName, role);
      if (hasQh) {
        // vid = vid_v_u32m4();  (the per-lane index j = 0..15.)
        mlir::Value vid = emitOpaqueCall(rewriter, loc, u32VecType,
                                         "__riscv_vid_v_u32m4",
                                         mlir::ValueRange{halfVl}, opName, role);
        if (qhShiftBase != 0)
          // high half: the 5th bit is bit (j+16) of qh.
          vid = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vadd_vx_u32m4",
                               mlir::ValueRange{vid, u32Lit(qhShiftBase), halfVl},
                               opName, role);
        // qhb = vmv_v_x_u32m4(qh);  sh = vsrl_vv(qhb, vid);  bit = sh & 1;
        // xh = bit << 4;  (== ((qh >> j) & 1) ? 0x10 : 0 -- the scalar's
        // ((qh>>j)<<4)&0x10 for the low half, ((qh>>(j+12)))&0x10 for the high half.)
        std::string bcastCallee = riscvIntrinsicName("vmv_v_x", 32, "m4", "u32");
        mlir::Value qhb = emitOpaqueCall(rewriter, loc, u32VecType, bcastCallee,
                                         mlir::ValueRange{qh, halfVl}, opName, role);
        mlir::Value sh = emitOpaqueCall(rewriter, loc, u32VecType,
                                        "__riscv_vsrl_vv_u32m4",
                                        mlir::ValueRange{qhb, vid, halfVl},
                                        opName, role);
        mlir::Value bit = emitOpaqueCall(rewriter, loc, u32VecType,
                                         "__riscv_vand_vx_u32m4",
                                         mlir::ValueRange{sh, u32Lit(1), halfVl},
                                         opName, role);
        mlir::Value xh = emitOpaqueCall(rewriter, loc, u32VecType,
                                        "__riscv_vsll_vx_u32m4",
                                        mlir::ValueRange{bit, u32Lit(4), halfVl},
                                        opName, role);
        n_u32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vor_vv_u32m4",
                               mlir::ValueRange{n_u32, xh, halfVl}, opName, role);
      }
      // n_i32 = reinterpret(n_u32);  (values in [0,31] -- bit-identical view.)
      mlir::Value n_i32 = emitOpaqueCall(
          rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
          mlir::ValueRange{n_u32}, opName, role);
      if (sub != 0)
        // n_i32 -= sub;  (the pre-scale bias: q4_0 -8, q5_0 -16.)
        n_i32 = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsub_vx_i32m4",
                               mlir::ValueRange{n_i32, i32Lit(sub), halfVl},
                               opName, role);
      // nF = vfcvt_f_x_v(n_i32);  (i32 -> f32, exact for |val| <= 31.)
      std::string cvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32");
      mlir::Value nF = emitOpaqueCall(rewriter, loc, f32VecType, cvtCallee,
                                      mlir::ValueRange{n_i32, halfVl}, opName, role);
      mlir::Value r;
      if (hasMin) {
        // r = vfmacc_vf(vfmv_v_f(m), d, nF);  (m + d*nF in ONE fused rounding ==
        // the CONTRACTED scalar `val*d + m`; the opponent autovec's it to vfmadd.)
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "nibble_scale_min"));
        std::string mBcastCallee = riscvIntrinsicName("vfmv_v_f", 32, "m4", "f32");
        mlir::Value macc = emitOpaqueCall(rewriter, loc, f32VecType, mBcastCallee,
                                          mlir::ValueRange{m, halfVl}, opName, role);
        r = emitOpaqueCall(rewriter, loc, f32VecType, "__riscv_vfmacc_vf_f32m4",
                           mlir::ValueRange{macc, d, nF, halfVl}, opName, role);
      } else {
        // r = vfmul_vf(nF, d);  (one f32 round-to-nearest-even multiply == scalar
        // `val*d`; q4_0/q5_0 have NO add/min => no fp-contraction ambiguity.)
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "nibble_scale"));
        std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, "m4", "f32");
        r = emitOpaqueCall(rewriter, loc, f32VecType, mulCallee,
                           mlir::ValueRange{nF, d, halfVl}, opName, role);
      }
      // vse32_v_f32m4(yStore, r, 16);  (the 16-float contiguous half-block store.)
      std::string vseCallee = riscvIntrinsicName("vse", 32, "m4", "f32");
      emitOpaqueCallVoid(rewriter, loc, vseCallee,
                         mlir::ValueRange{yStore, r, halfVl}, opName, role);
    };

    // Low nibbles: nlo = qs[j] & 0x0F  -> y[j..j+15].
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "nibble_decode_lo"));
    mlir::Value nlo = emitOpaqueCall(rewriter, loc, u8VecType,
                                     "__riscv_vand_vx_u8m1",
                                     mlir::ValueRange{qv, u8Lit(15), halfVl},
                                     opName, role);
    emitHalf(nlo, /*qhShiftBase=*/0, yb);

    // High nibbles: nhi = qs[j] >> 4  -> y[j+16..j+31].
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "nibble_decode_hi"));
    mlir::Value nhi = emitOpaqueCall(rewriter, loc, u8VecType,
                                     "__riscv_vsrl_vx_u8m1",
                                     mlir::ValueRange{qv, u8Lit(4), halfVl},
                                     opName, role);
    mlir::Value ybHi =
        rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(half));
    emitHalf(nhi, /*qhShiftBase=*/16, ybHi);
  }

  return mlir::success();
}

// The per-format OWNED REAL-VECTOR nibble leaves (emitDequantizeRowQ4_0VectorBody etc.)
// were RETIRED in phase-1 step (ii): the nibble dispatch (emitTypedDequantizeRowLoopBody)
// now READS the decode 8-tuple straight from the stamped decode_core descriptor and
// calls the SHARED emitDequantizeRowNibbleVectorBody directly, keyed on
// carrier_kind == "nibble4", so the per-format re-bake leaves carried no distinct
// behavior. The bare-int8 q8_0 leaf (emitDequantizeRowQ8_0VectorBody) stays separate
// ([K-10] leaf selection between the two already-separate leaves).

// ISSUE-001 reverse of the scalar emitDequantizeRowQ8_0BodyShared (whose owned vector
// intrinsic count is 0). The 32-lane block width is the FIXED q8_0 QK8_0 block
// geometry (NOT a tunable knob); the pipeline LMULs (i8m2 for the 32 int8 quants,
// i32m8/f32m8 for the 4x-widened 32-lane int->float pipeline) are DERIVED from that
// width, NOT literal knobs. Byte-exact-vs-ggml-reference dequantize_row_q8_0 by
// construction: the fp16 d seam is the SAME `(float)*(const _Float16 *)` read, the
// signed i8 quants sign-extend exactly, and vfmul_vf(qf, d) == the scalar `qs[j]*d`
// (a single f32 round-to-nearest-even multiply either way -- q8_0 has NO add/min, so
// there is NO fp-contraction ambiguity to break the byte-exact gate). Only the
// CONSTRUCTED path (emitTypedDequantizeRowLoopBody) routes here; the dispatch-wired
// monolith fallback stays on the scalar shared body (the iq3_xxs precedent).
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ8_0VectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    int64_t qk, int64_t stride, int64_t scaleOffset,
    int64_t quantOffset) const {
  if (qk != 32 || stride <= 0 || scaleOffset < 0 || quantOffset < 0)
    return rewriter.notifyMatchFailure(
        loc, "int8-scale emitter received an invalid typed plan");

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
  mlir::Type i8PtrType = emitc::PointerType::get(constI8Type);
  mlir::Type floatPtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
  // The 32-lane pipeline types: i8m2 holds the 32 int8 quants (VLEN>=128: m2 SEW8 >=
  // 32 lanes); the vsext_vf4 4x widening lands them in i32m8 (VLEN128 m8 SEW32 == 32
  // lanes exactly), then f32m8 for the convert + scale + store. All DERIVED from the
  // fixed 32-lane q8_0 block width.
  mlir::Type i8VecType = emitc::OpaqueType::get(ctx, "vint8m2_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m8_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // size_t nb = k / 32;  (ggml's `const int nb = k / qk`; k % qk == 0, no tail).
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1),
                                                /*bodyBuilder=*/nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    // const uint8_t *xb = x + ib*34;  float *yb = (float *)(y + ib*32);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "x_block"));
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
    mlir::Value scaleAddress =
        scaleOffset == 0
            ? xb
            : rewriter
                  .create<emitc::AddOp>(loc, inputPtrType, xb,
                                        sizeLit(scaleOffset))
                  .getResult();
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{scaleAddress}, opName, role,
                                   llvm::StringRef("fcvt.s.h"));

    // const int8_t *qs = (const int8_t *)(xb + 2);
    mlir::Value qsBaseRaw =
        rewriter.create<emitc::AddOp>(loc, inputPtrType, xb,
                                      sizeLit(quantOffset));
    mlir::Value qsBase =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, qsBaseRaw).getResult();

    // vint8m2_t qv = vle8_v_i8m2(qs, 32);  (the 32 signed int8 quants; a signed
    // load, so the widen sign-extends -- byte-identical to ggml's int8_t read.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "q8_load"));
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, "m2", "i8");
    mlir::Value qv = emitOpaqueCall(
        rewriter, loc, i8VecType, i8LoadCallee,
        mlir::ValueRange{qsBase, sizeLit(qk)}, opName, role);

    // int->float: q32 = vsext_vf4(qv); qf = vfcvt_f_x_v(q32).  (vl=32; int8 -> i32
    // sign-extend is exact, i32 -> f32 is exact for |q| <= 127.)
    mlir::Value q32 = emitOpaqueCall(
        rewriter, loc, i32VecType, "__riscv_vsext_vf4_i32m8",
        mlir::ValueRange{qv, sizeLit(qk)}, opName, role);
    std::string cvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, "m8", "f32");
    mlir::Value qf = emitOpaqueCall(
        rewriter, loc, f32VecType, cvtCallee,
        mlir::ValueRange{q32, sizeLit(qk)}, opName, role);

    // r = vfmul_vf(qf, d, 32);  (the runtime block scale; vfmul_vf(qf, d) == the
    // scalar `qs[j]*d` -- one f32 round-to-nearest-even multiply, byte-exact.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "q8_scale"));
    std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, "m8", "f32");
    mlir::Value r = emitOpaqueCall(
        rewriter, loc, f32VecType, mulCallee,
        mlir::ValueRange{qf, d, sizeLit(qk)}, opName, role);

    // vse32_v_f32m8(yb, r, 32);  (ONE contiguous 32-float store.)
    std::string vseCallee = riscvIntrinsicName("vse", 32, "m8", "f32");
    emitOpaqueCallVoid(rewriter, loc, vseCallee,
                       mlir::ValueRange{yb, r, sizeLit(qk)}, opName, role);
  }

  return mlir::success();
}

// ============================================================================
// The OWNED REAL-VECTOR K-quant super-block dequantize_row bodies (R线 §四.2, the
// K-quant fan-out over the q8_0/nibble non-grid precedent · de-lottery [L-8] ·
// ISSUE-001 reverse · closes the ISSUE-002 codegen-lottery exposure per format). Each
// mirrors the same integer quant decode as the retired scalar reference
// (byte-identical scale/min unpack + quant extraction) but folds the per-super-sub
// pipeline with OWNED __riscv_v intrinsics instead of ceding vectorization to the host
// autovec lottery. NO gather (K-quant is a bit-unpack, not a codebook lookup) -> NO
// HW-gather wall. Byte-exact-vs-ggml by construction (see the per-format notes).
// ============================================================================

// q4_K / q5_K (shared, keyed by isQ5): block_q4_K = { fp16 d@0, fp16 dmin@2,
// scales[12]@4, qs[128]@16 } stride 144; block_q5_K = { d@0, dmin@2, scales[12]@4,
// qh[32]@16, qs[128]@48 } stride 176. QK_K=256 = 4 super-sub-blocks of 64. Per
// super-sub jj (0..3): get_scale_min_k4 yields two 6-bit (sc,m) pairs; d1=d*sc0,
// ml1=dmin*m0 for the low 32 nibbles (qs&0xF), d2=d*sc1, ml2=dmin*m1 for the high 32
// (qs>>4). Each half is ONE 32-lane OWNED pipeline: vle8 (32 nibble bytes, u8m2) +
// vand/vsrl (the nibble plane) + [q5_K 5th-bit: per-lane (qh[l]>>(2*jj+half))&1 <<4,
// vsrl/vand/vsll/vor on the 32 contiguous qh bytes] + vzext_vf4 (-> u32m8) +
// vreinterpret (-> i32m8) + vfcvt (-> f32m8) + vfmv_v_f(ml)+vfmsac_vf(d1) (the FUSED
// d1*v - ml1, ONE rounding == the -ffp-contract=on scalar the opponent autovec's to
// vfmsub) + vse32. Byte-exact to dequantize_row_q4_K/q5_K by construction: the nibble
// (+5th-bit) integer value [0,31] is exact in f32, the scalar scale/min unpack is
// byte-identical, and the single fused mul-sub matches the contracted opponent.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ45KVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::KQuantScaleMinPlan &plan) const {
  // [K-10] STRUCTURAL TAG: this body realizes the KQuantScaleMin mechanism ONLY (the
  // dispatch guarantees plan.mechanism == KQuantScaleMin). The 5th-bit (q5_K) leaf flag
  // rides plan.scaleModel (Q5K), NOT the format name ([F-1]); the super-block geometry
  // (qk / stride / qs / dmin / scale-plane / qh offsets + strip lanes) rides the PLAN.
  const bool isQ5 = plan.scaleModel == ::weft::KQuantScaleModel::Q5K;
  const int64_t qk = plan.superBlockElements;
  const int64_t stride = plan.weightBlockStride;
  const int64_t qsOff = plan.quantByteOffset;
  const int64_t qhOff = plan.highBitByteOffset;  // q5_K only
  const int64_t dminOff = plan.minByteOffset;    // dmin @ scale-block + 2
  const int64_t subScaleOff = plan.subScaleByteOffset; // get_scale_min_k4 scales base
  const int64_t stripLanes = plan.stripLanes;

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m2_t");
  mlir::Type u16VecType = emitc::OpaqueType::get(ctx, "vuint16m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
  mlir::Type u8ScalarType = emitc::OpaqueType::get(ctx, "uint8_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType, std::to_string(v));
  };
  auto u8Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, u8ScalarType, std::to_string(v));
  };
  auto iAnd = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseAndOp>(loc, intType, a, b).getResult();
  };
  auto iOr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseOrOp>(loc, intType, a, b).getResult();
  };
  auto iShl = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, intType, a, b).getResult();
  };
  auto iShr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, a, b).getResult();
  };
  auto fMul = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, floatType, a, b).getResult();
  };
  auto i2f = [&](mlir::Value v) {
    return rewriter.create<emitc::CastOp>(loc, floatType, v).getResult();
  };
  auto mulSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  // Load *(base + off) as a uint8 byte widened to int (the proven pointer-advance +
  // subscript[0] idiom, byte-identical to ggml's `x[off]` promoted decode).
  auto loadU8Int = [&](mlir::Value base, int64_t off) -> mlir::Value {
    mlir::Value p = rewriter.create<emitc::AddOp>(loc, inputPtrType, base, sizeLit(off)).getResult();
    mlir::Value pc = rewriter.create<emitc::CastOp>(loc, u8PtrType, p).getResult();
    mlir::Value elem = rewriter.create<emitc::SubscriptOp>(
        loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pc), idxLit(0)).getResult();
    mlir::Value v = rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb = rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1), nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter.create<emitc::AddOp>(loc, inputPtrType, input, mulSz(ib, sizeLit(stride))).getResult();
    mlir::Value ybRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, output, mulSz(ib, sizeLit(qk))).getResult();
    mlir::Value yb = rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, isQ5 ? "q5_K_decode" : "q4_K_decode"));
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{xb}, opName, role, llvm::StringRef("fcvt.s.h"));
    mlir::Value dminAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(dminOff)).getResult();
    mlir::Value dmin = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dminAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    // get_scale_min_k4(j) for compile-time j (0..7), scales at plan.subScaleByteOffset
    // (== +4, byte-exact); the scale plane base rides the PLAN.
    auto sc = [&](int k) { return loadU8Int(xb, subScaleOff + k); };
    auto scaleMin = [&](int j, mlir::Value &scOut, mlir::Value &mOut) {
      if (j < 4) {
        scOut = iAnd(sc(j), intLit(63));
        mOut = iAnd(sc(j + 4), intLit(63));
      } else {
        scOut = iOr(iAnd(sc(j + 4), intLit(0x0F)), iShl(iShr(sc(j - 4), intLit(6)), intLit(4)));
        mOut = iOr(iShr(sc(j + 4), intLit(4)), iShl(iShr(sc(j), intLit(6)), intLit(4)));
      }
    };

    // q5_K: the 32 contiguous qh bytes (one per element l of the 64-element super-sub).
    mlir::Value qhv;
    if (isQ5) {
      mlir::Value qhPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qhOff)).getResult();
      mlir::Value qhU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qhPtr).getResult();
      qhv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m2", "u8"),
                           mlir::ValueRange{qhU8, sizeLit(stripLanes)}, opName, role);
    }

    // Emit one 32-lane half-super-sub pipeline: nibblePlane (u8m2) -> [q5 5th-bit at
    // bit `bitShift` of qh] -> u32m8 -> i32m8 -> f32m8 -> fused d*v - ml -> vse32.
    auto emitHalf = [&](mlir::Value nibblePlane, int64_t bitShift, mlir::Value scaleD,
                        mlir::Value minML, mlir::Value yStore) {
      mlir::Value plane = nibblePlane;
      if (isQ5) {
        mlir::Value sh = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m2",
                                        mlir::ValueRange{qhv, u8Lit(bitShift), sizeLit(stripLanes)}, opName, role);
        mlir::Value bit = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m2",
                                         mlir::ValueRange{sh, u8Lit(1), sizeLit(stripLanes)}, opName, role);
        mlir::Value hb = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsll_vx_u8m2",
                                        mlir::ValueRange{bit, u8Lit(4), sizeLit(stripLanes)}, opName, role);
        plane = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vor_vv_u8m2",
                               mlir::ValueRange{plane, hb, sizeLit(stripLanes)}, opName, role);
      }
      // W5 vfwcvt widen-shrink lever (byte-exact·u8 plane∈0..31<2^31·unsigned widen convert
      // == old signed vfcvt): vzext.vf4(u32m8)+vreinterpret+vfcvt → vzext.vf2(u16m4)+fused vfwcvt.
      mlir::Value n16 = emitOpaqueCall(rewriter, loc, u16VecType, "__riscv_vzext_vf2_u16m4",
                                       mlir::ValueRange{plane, sizeLit(stripLanes)}, opName, role);
      mlir::Value nF = emitOpaqueCall(rewriter, loc, f32VecType, "__riscv_vfwcvt_f_xu_v_f32m8",
                                      mlir::ValueRange{n16, sizeLit(stripLanes)}, opName, role);
      mlir::Value acc = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmv_v_f", 32, "m8", "f32"),
                                       mlir::ValueRange{minML, sizeLit(stripLanes)}, opName, role);
      mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, "__riscv_vfmsac_vf_f32m8",
                                     mlir::ValueRange{acc, scaleD, nF, sizeLit(stripLanes)}, opName, role);
      emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m8", "f32"),
                         mlir::ValueRange{yStore, r, sizeLit(stripLanes)}, opName, role);
    };

    // W5 scheduling lever (board-proven q4_K dequant 0.58→1.01, q5_K 1.16→2.21 @rvv, byte-exact):
    // TWO-PASS emit — hoist ALL scalar get_scale_min_k4 fp computation before the vector
    // pipelines. Interleaving the scalar fmul.s into the 8-pipeline vector region stalls each
    // vfmv.v.f/vfmsac.vf on a freshly-produced scalar on the in-order board; front-loading all
    // 8 scales lets the vector pipelines stream uninterrupted. Same values, only emission order
    // changes (byte-identical). This — not the vfwcvt widen-shrink — is the dominant lever.
    mlir::Value dLo[4], mlLo[4], dHi[4], mlHi[4];
    for (int64_t jj = 0; jj < 4; ++jj) {
      mlir::Value sc0, m0, sc1, m1v;
      scaleMin((int)(2 * jj), sc0, m0);
      scaleMin((int)(2 * jj + 1), sc1, m1v);
      dLo[jj] = fMul(d, i2f(sc0));
      mlLo[jj] = fMul(dmin, i2f(m0));
      dHi[jj] = fMul(d, i2f(sc1));
      mlHi[jj] = fMul(dmin, i2f(m1v));
    }
    for (int64_t jj = 0; jj < 4; ++jj) {
      mlir::Value qsPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff + jj * 32)).getResult();
      mlir::Value qsU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qsPtr).getResult();
      mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m2", "u8"),
                                      mlir::ValueRange{qsU8, sizeLit(stripLanes)}, opName, role);

      mlir::Value nlo = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m2",
                                       mlir::ValueRange{qv, u8Lit(15), sizeLit(stripLanes)}, opName, role);
      mlir::Value yLo = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(jj * 64)).getResult();
      emitHalf(nlo, 2 * jj, dLo[jj], mlLo[jj], yLo);

      mlir::Value nhi = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m2",
                                       mlir::ValueRange{qv, u8Lit(4), sizeLit(stripLanes)}, opName, role);
      mlir::Value yHi = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(jj * 64 + 32)).getResult();
      emitHalf(nhi, 2 * jj + 1, dHi[jj], mlHi[jj], yHi);
    }
  }
  return mlir::success();
}

// q2_K: block_q2_K = { scales[16]@0, qs[64]@16, fp16 d@80, fp16 dmin@82 } stride 84.
// QK_K=256. Per (nn 0..1, j 0..3, half 0..1) = 16 groups of 16: sc = scales[nn*8+2*j+half];
// dl = d*(sc&0xF), ml = dmin*(sc>>4); shift = 2*j. Each group is ONE 16-lane OWNED
// pipeline over 16 qs bytes: vle8 (u8m1) + vsrl_vx(shift)+vand_vx(3) (the 2-bit quant
// 0..3) + vzext_vf4 (-> u32m4) + vreinterpret (-> i32m4) + vfcvt (-> f32m4) +
// vfmv_v_f(ml)+vfmsac_vf(dl) (FUSED dl*q - ml) + vse32. Byte-exact by construction.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ2KVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::KQuantScaleMinPlan &plan) const {
  // [K-10] STRUCTURAL TAG: KQuantScaleMin ONLY (dispatch guarantees plan.mechanism ==
  // KQuantScaleMin). The super-block geometry (qk / stride / d / dmin / scale-plane / qs
  // offsets + strip lanes) rides the PLAN, NOT the format name ([F-1]).
  const int64_t qk = plan.superBlockElements;
  const int64_t stride = plan.weightBlockStride;
  const int64_t dOff = plan.scaleBlockByteOffset;      // fp16 d @ 80
  const int64_t dminOff = plan.minByteOffset;          // fp16 dmin @ 82 (== d + 2)
  const int64_t scalesOff = plan.subScaleByteOffset;   // packed 4-bit scale plane @ 0
  const int64_t qsOff = plan.quantByteOffset;          // qs base @ 16
  const int64_t stripLanes = plan.stripLanes;          // 16-lane sub-groups

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();
  mlir::Type outputPtrType = output.getType();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
  mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m4_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m4_t");
  mlir::Type u8ScalarType = emitc::OpaqueType::get(ctx, "uint8_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType, std::to_string(v));
  };
  auto u8Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, u8ScalarType, std::to_string(v));
  };
  auto iAnd = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseAndOp>(loc, intType, a, b).getResult();
  };
  auto iShr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, a, b).getResult();
  };
  auto fMul = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, floatType, a, b).getResult();
  };
  auto i2f = [&](mlir::Value v) {
    return rewriter.create<emitc::CastOp>(loc, floatType, v).getResult();
  };
  auto mulSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  auto loadU8Int = [&](mlir::Value base, int64_t off) -> mlir::Value {
    mlir::Value p = rewriter.create<emitc::AddOp>(loc, inputPtrType, base, sizeLit(off)).getResult();
    mlir::Value pc = rewriter.create<emitc::CastOp>(loc, u8PtrType, p).getResult();
    mlir::Value elem = rewriter.create<emitc::SubscriptOp>(
        loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pc), idxLit(0)).getResult();
    mlir::Value v = rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb = rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1), nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter.create<emitc::AddOp>(loc, inputPtrType, input, mulSz(ib, sizeLit(stride))).getResult();
    mlir::Value ybRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, output, mulSz(ib, sizeLit(qk))).getResult();
    mlir::Value yb = rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "q2_K_decode"));
    mlir::Value dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(dOff)).getResult();
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{dAddr}, opName, role, llvm::StringRef("fcvt.s.h"));
    mlir::Value dminAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(dminOff)).getResult();
    mlir::Value dmin = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dminAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    for (int64_t nn = 0; nn < 2; ++nn) {
      int64_t qBlk = qsOff + nn * 32;
      int64_t outBlk = nn * 128;
      for (int64_t j = 0; j < 4; ++j) {
        int64_t shift = 2 * j;
        int64_t is0 = nn * 8 + j * 2;
        int64_t outJ = outBlk + j * 32;
        for (int64_t half = 0; half < 2; ++half) {
          mlir::Value scv = loadU8Int(xb, scalesOff + is0 + half);
          mlir::Value dl = fMul(d, i2f(iAnd(scv, intLit(0xF))));
          mlir::Value ml = fMul(dmin, i2f(iShr(scv, intLit(4))));
          int64_t qHalf = qBlk + half * 16;
          int64_t outHalf = outJ + half * 16;

          mlir::Value qPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qHalf)).getResult();
          mlir::Value qU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qPtr).getResult();
          mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                          mlir::ValueRange{qU8, sizeLit(stripLanes)}, opName, role);
          mlir::Value shd = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                           mlir::ValueRange{qv, u8Lit(shift), sizeLit(stripLanes)}, opName, role);
          mlir::Value q2 = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                          mlir::ValueRange{shd, u8Lit(3), sizeLit(stripLanes)}, opName, role);
          mlir::Value q32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                           mlir::ValueRange{q2, sizeLit(stripLanes)}, opName, role);
          mlir::Value qi = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                          mlir::ValueRange{q32}, opName, role);
          mlir::Value qF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                          mlir::ValueRange{qi, sizeLit(stripLanes)}, opName, role);
          mlir::Value acc = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmv_v_f", 32, "m4", "f32"),
                                           mlir::ValueRange{ml, sizeLit(stripLanes)}, opName, role);
          mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, "__riscv_vfmsac_vf_f32m4",
                                         mlir::ValueRange{acc, dl, qF, sizeLit(stripLanes)}, opName, role);
          mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outHalf)).getResult();
          emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                             mlir::ValueRange{yStore, r, sizeLit(stripLanes)}, opName, role);
        }
      }
    }
  }
  return mlir::success();
}

// q3_K: block_q3_K = { hmask[32]@0, qs[64]@32, scales[12]@96, fp16 d@108 } stride 110.
// QK_K=256. The 6-bit signed scales come from the aux kmask 12-byte shuffle (q3Scale(is),
// is 0..15, compile-time byte arithmetic). Per (nn 0..1, j 0..3, half 0..1) = 16 groups
// of 16: is = nn*8+2*j+half; dl = d*(q3Scale(is)-32); shift = 2*j; mbit = nn*4+j. Each
// group is ONE 16-lane OWNED pipeline: vle8 qs + vle8 hmask (u8m1) + the 2-bit quant
// (vsrl(shift)+vand(3)) + the hmask high term ((hm>>mbit)&1 -> (1-bit)<<2) in i32m4 ->
// qdec = qbits - term -> vfcvt -> vfmul_vf(dl) (SINGLE mul, NO min -> no fp-contraction
// ambiguity) -> vse32. Byte-exact by construction.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ3KVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::KQuantScaleMinPlan &plan) const {
  // [K-10] STRUCTURAL TAG: KQuantScaleMin ONLY (dispatch guarantees plan.mechanism ==
  // KQuantScaleMin). The super-block geometry (qk / stride / d / aux-scale-plane / hmask
  // / qs offsets + strip lanes) rides the PLAN, NOT the format name ([F-1]). q3_K folds
  // with a SINGLE mul and carries NO min (plan.hasMin == false).
  const int64_t qk = plan.superBlockElements;
  const int64_t stride = plan.weightBlockStride;
  const int64_t dOff = plan.scaleBlockByteOffset;      // fp16 d @ 108
  const int64_t scalesOff = plan.subScaleByteOffset;   // aux 6-bit scale plane @ 96
  const int64_t hmaskOff = plan.highBitByteOffset;     // hmask high-bit plane @ 0
  const int64_t qsOff = plan.quantByteOffset;          // qs base @ 32
  const int64_t stripLanes = plan.stripLanes;          // 16-lane sub-groups

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();
  mlir::Type outputPtrType = output.getType();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
  mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m4_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m4_t");
  mlir::Type u8ScalarType = emitc::OpaqueType::get(ctx, "uint8_t");
  mlir::Type i32ScalarType = emitc::OpaqueType::get(ctx, "int32_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType, std::to_string(v));
  };
  auto u8Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, u8ScalarType, std::to_string(v));
  };
  auto i32Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, i32ScalarType, std::to_string(v));
  };
  auto iAnd = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseAndOp>(loc, intType, a, b).getResult();
  };
  auto iOr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseOrOp>(loc, intType, a, b).getResult();
  };
  auto iShl = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, intType, a, b).getResult();
  };
  auto iShr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, a, b).getResult();
  };
  auto iSub = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::SubOp>(loc, intType, a, b).getResult();
  };
  auto fMul = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, floatType, a, b).getResult();
  };
  auto i2f = [&](mlir::Value v) {
    return rewriter.create<emitc::CastOp>(loc, floatType, v).getResult();
  };
  auto mulSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  auto loadU8Int = [&](mlir::Value base, int64_t off) -> mlir::Value {
    mlir::Value p = rewriter.create<emitc::AddOp>(loc, inputPtrType, base, sizeLit(off)).getResult();
    mlir::Value pc = rewriter.create<emitc::CastOp>(loc, u8PtrType, p).getResult();
    mlir::Value elem = rewriter.create<emitc::SubscriptOp>(
        loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pc), idxLit(0)).getResult();
    mlir::Value v = rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb = rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1), nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter.create<emitc::AddOp>(loc, inputPtrType, input, mulSz(ib, sizeLit(stride))).getResult();
    mlir::Value ybRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, output, mulSz(ib, sizeLit(qk))).getResult();
    mlir::Value yb = rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "q3_K_decode"));
    mlir::Value dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(dOff)).getResult();
    mlir::Value dAll = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    // The derived 6-bit scale for compile-time is (0..15) from the 12 packed scale
    // bytes at plan.subScaleByteOffset (== +96, ggml aux kmask1/kmask2 shuffle,
    // byte-exact to the scalar decode); the scale plane base rides the PLAN.
    auto q3Scale = [&](int is) -> mlir::Value {
      int group = is / 4, b = is % 4;
      auto sb = [&](int k) { return loadU8Int(xb, scalesOff + k); };
      mlir::Value lowNib, hiPart;
      if (group == 0) {
        lowNib = iAnd(sb(b), intLit(0x0F));
        hiPart = iShl(iAnd(sb(8 + b), intLit(0x03)), intLit(4));
      } else if (group == 1) {
        lowNib = iAnd(sb(4 + b), intLit(0x0F));
        hiPart = iShl(iAnd(iShr(sb(8 + b), intLit(2)), intLit(0x03)), intLit(4));
      } else if (group == 2) {
        lowNib = iAnd(iShr(sb(b), intLit(4)), intLit(0x0F));
        hiPart = iShl(iAnd(iShr(sb(8 + b), intLit(4)), intLit(0x03)), intLit(4));
      } else {
        lowNib = iAnd(iShr(sb(4 + b), intLit(4)), intLit(0x0F));
        hiPart = iShl(iAnd(iShr(sb(8 + b), intLit(6)), intLit(0x03)), intLit(4));
      }
      return iOr(lowNib, hiPart);
    };

    for (int64_t nn = 0; nn < 2; ++nn) {
      int64_t qBlk = qsOff + nn * 32;
      int64_t outBlk = nn * 128;
      for (int64_t j = 0; j < 4; ++j) {
        int64_t shift = 2 * j;
        int64_t mbit = nn * 4 + j;
        int64_t outJ = outBlk + j * 32;
        for (int64_t half = 0; half < 2; ++half) {
          int is = (int)(nn * 8 + 2 * j + half);
          mlir::Value dl = fMul(dAll, i2f(iSub(q3Scale(is), intLit(32))));
          int64_t qHalf = qBlk + half * 16;
          int64_t hmHalf = hmaskOff + half * 16;
          int64_t outHalf = outJ + half * 16;

          mlir::Value qPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qHalf)).getResult();
          mlir::Value qU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qPtr).getResult();
          mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                          mlir::ValueRange{qU8, sizeLit(stripLanes)}, opName, role);
          mlir::Value hmPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(hmHalf)).getResult();
          mlir::Value hmU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, hmPtr).getResult();
          mlir::Value hmv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                           mlir::ValueRange{hmU8, sizeLit(stripLanes)}, opName, role);

          mlir::Value shd = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                           mlir::ValueRange{qv, u8Lit(shift), sizeLit(stripLanes)}, opName, role);
          mlir::Value qbits8 = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                              mlir::ValueRange{shd, u8Lit(3), sizeLit(stripLanes)}, opName, role);
          mlir::Value hmsh = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                            mlir::ValueRange{hmv, u8Lit(mbit), sizeLit(stripLanes)}, opName, role);
          mlir::Value bit8 = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                            mlir::ValueRange{hmsh, u8Lit(1), sizeLit(stripLanes)}, opName, role);

          mlir::Value qbits32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                               mlir::ValueRange{qbits8, sizeLit(stripLanes)}, opName, role);
          mlir::Value qbitsI = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                              mlir::ValueRange{qbits32}, opName, role);
          mlir::Value bit32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                             mlir::ValueRange{bit8, sizeLit(stripLanes)}, opName, role);
          mlir::Value bitI = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                            mlir::ValueRange{bit32}, opName, role);
          // term = (1 - bit) << 2 ; qdec = qbits - term.
          mlir::Value oneMinus = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vrsub_vx_i32m4",
                                                mlir::ValueRange{bitI, i32Lit(1), sizeLit(stripLanes)}, opName, role);
          mlir::Value term = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsll_vx_i32m4",
                                            mlir::ValueRange{oneMinus, i32Lit(2), sizeLit(stripLanes)}, opName, role);
          mlir::Value qdec = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsub_vv_i32m4",
                                            mlir::ValueRange{qbitsI, term, sizeLit(stripLanes)}, opName, role);
          mlir::Value qF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                          mlir::ValueRange{qdec, sizeLit(stripLanes)}, opName, role);
          mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, "m4", "f32"),
                                         mlir::ValueRange{qF, dl, sizeLit(stripLanes)}, opName, role);
          mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outHalf)).getResult();
          emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                             mlir::ValueRange{yStore, r, sizeLit(stripLanes)}, opName, role);
        }
      }
    }
  }
  return mlir::success();
}

// q6_K: block_q6_K = { ql[128]@0, qh[64]@128, int8 scales[16]@192, fp16 d@208 } stride
// 210. QK_K=256. y = d*sc[is]*q where q = ((ql nibble) | ((qh 2 bits)<<4)) - 32. Within
// each nn's 32-lane span the scale index is = l/16 splits at 16, so the span is emitted
// as TWO 16-lane groups (is 0..1). Per (nn 0..1, is 0..1, t 0..3=q1..q4) = 16 groups of
// 16: sc = scales[192 + nn*8 + is + 2*t] (SIGNED int8); dsc = d*sc. Each group is ONE
// 16-lane OWNED pipeline: vle8 ql + vle8 qh (u8m1) + the nibble (lo t<2, hi t>=2) + the
// qh 2-bit high term ((qh>>(2*t))&3 <<4) -> combined 0..63 -> vsub 32 in i32m4 -> vfcvt
// -> vfmul_vf(dsc) (SINGLE mul == the scalar `(d*sc)*q`, no fp-contraction) -> vse32.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ6KVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::KQuantScaleMinPlan &plan) const {
  // [K-10] STRUCTURAL TAG: KQuantScaleMin ONLY (dispatch guarantees plan.mechanism ==
  // KQuantScaleMin). The super-block geometry (qk / stride / d / int8-scale-plane / qh
  // / ql offsets + strip lanes) rides the PLAN, NOT the format name ([F-1]). q6_K folds
  // with a SINGLE mul and carries NO min (plan.hasMin == false).
  const int64_t qk = plan.superBlockElements;
  const int64_t stride = plan.weightBlockStride;
  const int64_t dOff = plan.scaleBlockByteOffset;      // fp16 d @ 208
  const int64_t scalesOff = plan.subScaleByteOffset;   // signed-int8 scale plane @ 192
  const int64_t qhBaseOff = plan.highBitByteOffset;    // qh high-bit plane @ 128
  const int64_t qlOffBase = plan.quantByteOffset;      // ql base @ 0
  const int64_t stripLanes = plan.stripLanes;          // 16-lane sub-groups

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();
  mlir::Type outputPtrType = output.getType();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
  mlir::Type i8PtrType = emitc::PointerType::get(constI8Type);
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
  mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m4_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m4_t");
  mlir::Type u8ScalarType = emitc::OpaqueType::get(ctx, "uint8_t");
  mlir::Type i32ScalarType = emitc::OpaqueType::get(ctx, "int32_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType, std::to_string(v));
  };
  auto u8Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, u8ScalarType, std::to_string(v));
  };
  auto i32Lit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, i32ScalarType, std::to_string(v));
  };
  auto fMul = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, floatType, a, b).getResult();
  };
  auto i2f = [&](mlir::Value v) {
    return rewriter.create<emitc::CastOp>(loc, floatType, v).getResult();
  };
  auto mulSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  // Load a SIGNED int8 scale byte at compile-time off (sign-extends to int).
  auto loadI8Int = [&](mlir::Value baseI8, int64_t off) -> mlir::Value {
    mlir::Value p = rewriter.create<emitc::AddOp>(loc, i8PtrType, baseI8, sizeLit(off)).getResult();
    mlir::Value elem = rewriter.create<emitc::SubscriptOp>(
        loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(p), idxLit(0)).getResult();
    mlir::Value v = rewriter.create<emitc::LoadOp>(loc, constI8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb = rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1), nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter.create<emitc::AddOp>(loc, inputPtrType, input, mulSz(ib, sizeLit(stride))).getResult();
    mlir::Value xbI8 = rewriter.create<emitc::CastOp>(loc, i8PtrType, xb).getResult();
    mlir::Value ybRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, output, mulSz(ib, sizeLit(qk))).getResult();
    mlir::Value yb = rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "q6_K_decode"));
    mlir::Value dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(dOff)).getResult();
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{dAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    for (int64_t nn = 0; nn < 2; ++nn) {
      int64_t qlBlk = qlOffBase + nn * 64;
      int64_t qhBlk = qhBaseOff + nn * 32;
      int64_t scBlk = scalesOff + nn * 8;
      int64_t outBlk = nn * 128;
      for (int64_t is = 0; is < 2; ++is) {
        int64_t lbase = is * 16;
        for (int64_t t = 0; t < 4; ++t) {
          bool nibLo = (t < 2);
          int64_t qlOff = qlBlk + ((t & 1) ? 32 : 0) + lbase; // q1/q3 use ql@0, q2/q4 use ql@32
          int64_t qhShift = 2 * t;
          int64_t scOff = scBlk + is + 2 * t;
          int64_t outOff = outBlk + t * 32 + lbase;

          mlir::Value dsc = fMul(d, i2f(loadI8Int(xbI8, scOff)));

          mlir::Value qlPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qlOff)).getResult();
          mlir::Value qlU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qlPtr).getResult();
          mlir::Value qlv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                           mlir::ValueRange{qlU8, sizeLit(stripLanes)}, opName, role);
          mlir::Value qhPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qhBlk + lbase)).getResult();
          mlir::Value qhU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qhPtr).getResult();
          mlir::Value qhv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                           mlir::ValueRange{qhU8, sizeLit(stripLanes)}, opName, role);

          mlir::Value nib;
          if (nibLo)
            nib = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                 mlir::ValueRange{qlv, u8Lit(0x0F), sizeLit(stripLanes)}, opName, role);
          else
            nib = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                 mlir::ValueRange{qlv, u8Lit(4), sizeLit(stripLanes)}, opName, role);
          mlir::Value qhsh = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                            mlir::ValueRange{qhv, u8Lit(qhShift), sizeLit(stripLanes)}, opName, role);
          mlir::Value qhbits = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                              mlir::ValueRange{qhsh, u8Lit(3), sizeLit(stripLanes)}, opName, role);
          mlir::Value qhhi = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsll_vx_u8m1",
                                            mlir::ValueRange{qhbits, u8Lit(4), sizeLit(stripLanes)}, opName, role);
          mlir::Value combined = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vor_vv_u8m1",
                                                mlir::ValueRange{nib, qhhi, sizeLit(stripLanes)}, opName, role);
          mlir::Value c32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                           mlir::ValueRange{combined, sizeLit(stripLanes)}, opName, role);
          mlir::Value ci = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                          mlir::ValueRange{c32}, opName, role);
          mlir::Value qi = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsub_vx_i32m4",
                                          mlir::ValueRange{ci, i32Lit(32), sizeLit(stripLanes)}, opName, role);
          mlir::Value qF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                          mlir::ValueRange{qi, sizeLit(stripLanes)}, opName, role);
          mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, "m4", "f32"),
                                         mlir::ValueRange{qF, dsc, sizeLit(stripLanes)}, opName, role);
          mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outOff)).getResult();
          emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                             mlir::ValueRange{yStore, r, sizeLit(stripLanes)}, opName, role);
        }
      }
    }
  }
  return mlir::success();
}

// The per-format CONSTRUCTED dequantize_row decode leaf for the QK_K=256 IQ
// grid-table super-block family (iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s): a thin
// FORWARDER to the SAME hand-written grid-decode the dispatch-wired monolith fallback
// used by the retired scalar reference. Each IQ grid leaf emits its canonical grid
// + sign-plane table decls as
// function-local statics (the SAME emitIQ2XXSCanonicalGridTableDecl / ...Signs64 /
// ...Signs256 / ...Ksigns anchors the block-dot vec_dot lowering renders) then a
// scalar AoS super-block loop -- there is NO op-attribute dependency (the signs64 /
// grid-table planes are DERIVED at emit, not carried as op-attrs), so the leaf is
// self-contained and CLEANLY constructible. Because BOTH the monolith fallback AND
// this constructed lowering emit the IQ grid decode from the SAME code, the two are
// byte-exact by construction (modulo only the source-op provenance token threaded
// through opName/role) -- there is NO duplicated IQ grid decode leaf to drift.
// Streaming sibling of emitDequantizeRowKQuantBodyShared; no reduction / no
// accumulator.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowIQGridBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::GridLookupPlan &plan) const {
  // [K-10] STRUCTURAL TAG: this body realizes the GridLookup mechanism ONLY (the dispatch
  // guarantees plan.mechanism == GridLookup). The per-format owned real-vector body is
  // selected by plan.leaf (a leaf SELECTION between the FIVE already-separate gather-free
  // bodies, sanctioned by [K-10]), NOT the format name ([F-1]); the grid ENTRY byte-width
  // g-axis geometry rides plan.entryLanes (律2: retired into the codebook_entry_lanes
  // descriptor, NOT baked). The four int-grid leaves fail-CLOSE if the descriptor is absent
  // (plan.hasEntryLanes false -- no value_or self-supply to the g-axis); iq3_xxs derives its
  // geometry internally and does not require it. Byte-exact reproduce-current: plan.leaf +
  // plan.entryLanes reproduce the retired (format, codebookEntryLanes) dispatch exactly.
  switch (plan.leaf) {
  // iq3_xxs is the FIRST cell of the dequant true-vector emitter (PR-31): it lowers to the
  // OWNED real-vector body (vluxei16 grid gather + sign fold + vfcvt + vfmul + vse32).
  case ::weft::GridDecodeLeaf::Iq3Xxs:
    return emitDequantizeRowIQ3XXSVectorBody(rewriter, loc, input, output, avlArg,
                                             sizeType, opName, role);
  // iq3_s (R5.1-C grid family de-lottery): the grid-of-4 EXPLICIT-SIGNS sibling lowers to
  // its OWNED narrow-per-entry real-vector body (gather-free, board-proven).
  case ::weft::GridDecodeLeaf::Iq3S:
    if (!plan.hasEntryLanes)
      return rewriter.notifyMatchFailure(
          loc, "iq3_s owned grid dequant body requires the codebook_entry_lanes "
               "descriptor (grid ENTRY byte-width g-axis geometry) stamped by the "
               "dequant-stream front door; it must NOT be baked into the mechanism "
               "body and is never value_or self-supplied");
    return emitDequantizeRowIQ3SVectorBody(rewriter, loc, input, output, avlArg,
                                           sizeType, opName, role, plan.entryLanes);
  // iq2_xs (R5.1-D 扩 iq2 面 · grid family de-lottery): the grid-of-8 (int64, 512-entry)
  // sibling lowers to its OWNED narrow-per-entry real-vector body (gather-free -- the
  // deployed scalar forwarder's clang autovec exploded to 32 vluxei / 96 vslidedown).
  case ::weft::GridDecodeLeaf::Iq2Xs:
    if (!plan.hasEntryLanes)
      return rewriter.notifyMatchFailure(
          loc, "iq2_xs owned grid dequant body requires the codebook_entry_lanes "
               "descriptor (grid ENTRY byte-width g-axis geometry) stamped by the "
               "dequant-stream front door; it must NOT be baked into the mechanism "
               "body and is never value_or self-supplied");
    return emitDequantizeRowIQ2XSVectorBody(rewriter, loc, input, output, avlArg,
                                            sizeType, opName, role, plan.entryLanes);
  // iq2_xxs (W5 grid family de-lottery): the grid-of-8 (int64, 256-entry) sibling with the
  // aux-packed ksigns selectors lowers to its OWNED narrow-per-entry real-vector body.
  case ::weft::GridDecodeLeaf::Iq2Xxs:
    if (!plan.hasEntryLanes)
      return rewriter.notifyMatchFailure(
          loc, "iq2_xxs owned grid dequant body requires the codebook_entry_lanes "
               "descriptor (grid ENTRY byte-width g-axis geometry) stamped by the "
               "dequant-stream front door; it must NOT be baked into the mechanism "
               "body and is never value_or self-supplied");
    return emitDequantizeRowIQ2XXSVectorBody(rewriter, loc, input, output, avlArg,
                                             sizeType, opName, role, plan.entryLanes);
  // iq2_s (W5 grid family de-lottery): the grid-of-8 (int64, 1024-entry) sibling whose
  // 8-bit sign bytes index the universal signs256 plane lowers to its OWNED body.
  case ::weft::GridDecodeLeaf::Iq2S:
    if (!plan.hasEntryLanes)
      return rewriter.notifyMatchFailure(
          loc, "iq2_s owned grid dequant body requires the codebook_entry_lanes "
               "descriptor (grid ENTRY byte-width g-axis geometry) stamped by the "
               "dequant-stream front door; it must NOT be baked into the mechanism "
               "body and is never value_or self-supplied");
    return emitDequantizeRowIQ2SVectorBody(rewriter, loc, input, output, avlArg,
                                           sizeType, opName, role, plan.entryLanes);
  }
  llvm_unreachable("GridLookupPlan leaf not handled");
}

// The per-format CONSTRUCTED dequantize_row decode leaf for the ternary-grid extended
// formats: the ternary iq1s_grid leaves iq1_s (fp16 d + qh scale + delta) / iq1_m
// (reconstructed packed iq1m_scale + per-group delta, NO fp16 d), and the tq1_0/tq2_0
// base-3 / 2-bit ternary super-blocks. (The iq4_nl/iq4_xs/mxfp4/nvfp4 codebook leaves --
// a DIFFERENT mechanism, CodebookGather -- are dispatched UPSTREAM via CodebookGatherPlan
// straight to emitDequantizeRowCodebookVectorBody; they no longer reach this forwarder.
// [K-10]: TernaryDecode and CodebookGather are structurally distinct mechanisms.) A thin
// FORWARDER to the SAME hand-written extended decode the dispatch-wired monolith fallback
// used by the retired scalar reference.
// Each leaf emits its grid table as function-local statics (the SAME
// emitIQ1SCanonicalGridTableDecl / emitIQ1MCanonicalGridTableDecl anchors the block-dot
// vec_dot lowerings render) then a scalar AoS block loop -- there is NO op-attribute
// dependency (the grid / ternary planes are DERIVED at emit, not carried as op-attrs), so
// the leaf is self-contained and CLEANLY constructible. Because BOTH the monolith fallback
// AND this constructed lowering emit the decode from the SAME code, the two are byte-exact
// by construction (modulo only the source-op provenance token threaded through
// opName/role). Streaming sibling of emitDequantizeRowIQGridBodyShared; no reduction / no
// accumulator.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowTernaryDecodeBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::TernaryDecodePlan &plan) const {
  // [K-10] STRUCTURAL TAG: this body realizes the TernaryDecode mechanism ONLY (the
  // dispatch guarantees plan.mechanism == TernaryDecode). Ternary is SPLIT OUT of grid --
  // it has its own plan and does NOT consult the grid registry (grid != ternary restored
  // for the dequant-row head). The per-format owned body is selected by plan.leaf (a leaf
  // SELECTION between the FOUR already-separate bodies, [K-10]), NOT the format name
  // ([F-1]); the iq1 grid leaves' ENTRY byte-width g-axis geometry rides plan.entryLanes
  // (律2) and fail-CLOSES if absent. Byte-exact reproduce-current: plan.leaf +
  // plan.entryLanes reproduce the retired (format, codebookEntryLanes) dispatch exactly.
  switch (plan.leaf) {
  // The tq1_0/tq2_0 base-3 / 2-bit ternary super-blocks lower to the OWNED ternary
  // ARITHMETIC vector body (B线批3 ternary de-lottery · [L-8] · closes the ISSUE-002
  // codegen-lottery for these formats). NO codebook table and NO gather -- the ternary
  // {-1,0,1} value is decoded by pure arithmetic (2-bit shift/mask; base-3 * pow3). The
  // isTq1 leaf discriminator selects the two ALREADY-SEPARATE tq shapes (no format key).
  case ::weft::TernaryDecodeLeaf::Tq1_0:
  case ::weft::TernaryDecodeLeaf::Tq2_0:
    return emitDequantizeRowTernaryVectorBody(
        rewriter, loc, input, output, avlArg, sizeType, opName, role,
        /*isTq1=*/plan.leaf == ::weft::TernaryDecodeLeaf::Tq1_0);
  // iq1_m FANS OUT to its OWNED narrow-per-entry real-vector body (R5.1-D grid family
  // de-lottery [L-8] · ISSUE-001 reverse). Unlike the iq2_xs/iq3_s flip, iq1_m's deployed
  // scalar forwarder is ALREADY gather-free after clang -O3 autovec, so this OWNED body is a
  // de-lottery robustness hardening at PARITY throughput (lever-N/A honest-null). The grid
  // ENTRY byte-width (g-axis geometry) rides plan.entryLanes (律2); fail closed if the front
  // door did not stamp it (no value_or self-supply to the g-axis).
  case ::weft::TernaryDecodeLeaf::Iq1M:
    if (!plan.hasEntryLanes)
      return rewriter.notifyMatchFailure(
          loc, "iq1_m owned grid dequant body requires the codebook_entry_lanes "
               "descriptor (grid ENTRY byte-width g-axis geometry) stamped by the "
               "dequant-stream front door; it must NOT be baked into the mechanism "
               "body and is never value_or self-supplied");
    return emitDequantizeRowIQ1MVectorBody(rewriter, loc, input, output, avlArg,
                                           sizeType, opName, role, plan.entryLanes);
  // iq1_s FANS OUT (W5 grid family de-lottery · the iq1_m ternary sibling completing the
  // ternary-grid flip): the SIGNED ternary iq1s_grid (2048-entry) sibling -- fp16 d read
  // DIRECTLY, ONE scale + ONE delta per group -- lowers to its OWNED narrow-per-entry
  // real-vector body (gather-free; vfadd(delta)+vfmul(dl), NO fused vfmacc, so the two
  // roundings match ggml). Same g-axis descriptor read + fail-closed contract as iq1_m.
  case ::weft::TernaryDecodeLeaf::Iq1S:
    if (!plan.hasEntryLanes)
      return rewriter.notifyMatchFailure(
          loc, "iq1_s owned grid dequant body requires the codebook_entry_lanes "
               "descriptor (grid ENTRY byte-width g-axis geometry) stamped by the "
               "dequant-stream front door; it must NOT be baked into the mechanism "
               "body and is never value_or self-supplied");
    return emitDequantizeRowIQ1SVectorBody(rewriter, loc, input, output, avlArg,
                                           sizeType, opName, role, plan.entryLanes);
  }
  llvm_unreachable("TernaryDecodePlan leaf not handled");
}

// ============================================================================
// The OWNED REAL-VECTOR tiny-codebook (16-entry) dequantize_row body (B线批2, the FP4 /
// non-linear codebook fan-out over the q8_0/nibble non-grid precedent · de-lottery [L-8]
// · ISSUE-001 reverse · closes the ISSUE-002 codegen-lottery exposure per format). The
// 16-entry int8 codebook is broadcast into one selected i8{mf2|m1|m2} vreg ONCE;
// every group's two nibble index lanes are gathered through it (a REGISTER-RESIDENT gather,
// NOT a vluxei memory gather -> NO HW-gather wall), sign-extended, int->float, scaled by
// the per-group float scale in ONE vfmul (== ggml's single `d*kv` mul -> no
// fp-contraction ambiguity), and stored. The integer nibble/codebook/scale decode
// mirrors the scalar reference byte-for-byte. Byte-exact to
// dequantize_row_{mxfp4,nvfp4,iq4_nl,iq4_xs} by construction.
// ============================================================================
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowCodebookVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    const ::weft::CodebookGatherPlan &plan) const {
  // [K-10] STRUCTURAL TAG: this body realizes the CodebookGather mechanism ONLY (the
  // dispatch guarantees plan.mechanism == CodebookGather). The per-format scale-decode
  // leaf is selected by plan.scaleModel (the block-type's structural scale ABI), NOT the
  // format name ([F-1]): E8M0 (mxfp4) / fp16 (iq4_nl) / UE4M3 (nvfp4) / signed-6 (iq4_xs).
  const bool isMx = plan.scaleModel == ::weft::CodebookScaleModel::E8M0SharedExp;
  const bool isNl = plan.scaleModel == ::weft::CodebookScaleModel::Fp16Flat;
  const bool isNv = plan.scaleModel == ::weft::CodebookScaleModel::UE4M3SubBlock;
  const bool isXs = plan.scaleModel == ::weft::CodebookScaleModel::Signed6SuperBlock;
  // Super-block geometry carried by the final formula plan (byte-exact ggml
  // block_qX AoS facts, NOT knobs): qk / stride / base qs offset ride the PLAN.
  const int64_t qk = plan.superBlockElements;
  const int64_t stride = plan.weightBlockStride;
  const int64_t qsBase = plan.codebookByteOffset;
  const int64_t stripLanes = plan.stripLanes;
  // The i8 codebook-gather anchor LMUL rides the selected PLAN; the widened i32/f32
  // chain is mechanically DERIVED from it
  // via deriveWideningChain (the SAME single-source-of-truth the vec_dot codebook body
  // uses), so a plan.loadLMUL change re-shapes every codebook intrinsic. i8 -> i32 (vf4)
  // is two LMUL doublings, so the widened rung is the chain's l32.
  llvm::StringRef coreLmul = plan.loadLMUL;
  WideningChain wideningChain = deriveWideningChain(coreLmul);
  llvm::StringRef wideLmul = wideningChain.l32;

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
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
  mlir::Type u8VecType =
      emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
  mlir::Type i8VecType =
      emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
  mlir::Type i32VecType =
      emitc::OpaqueType::get(ctx, ("vint32" + wideLmul + "_t").str());
  mlir::Type f32VecType =
      emitc::OpaqueType::get(ctx, ("vfloat32" + wideLmul + "_t").str());
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
  auto uLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
  auto idxLit = [&](int64_t v) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType, std::to_string(v));
  };
  auto floatLit = [&](llvm::StringRef s) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, floatType, s);
  };
  auto iAnd = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseAndOp>(loc, intType, a, b).getResult();
  };
  auto iOr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseOrOp>(loc, intType, a, b).getResult();
  };
  auto iShl = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, intType, a, b).getResult();
  };
  auto iShr = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, a, b).getResult();
  };
  auto iSub = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::SubOp>(loc, intType, a, b).getResult();
  };
  auto fMul = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, floatType, a, b).getResult();
  };
  auto i2f = [&](mlir::Value v) {
    return rewriter.create<emitc::CastOp>(loc, floatType, v).getResult();
  };
  auto mulSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  // Load *(base + off) as a uint8 byte widened to int (the proven pointer-advance +
  // subscript[0] idiom, byte-identical to ggml's `x[off]` promoted decode).
  auto loadU8Int = [&](mlir::Value base, int64_t off) -> mlir::Value {
    mlir::Value p = rewriter.create<emitc::AddOp>(loc, inputPtrType, base, sizeLit(off)).getResult();
    mlir::Value pc = rewriter.create<emitc::CastOp>(loc, u8PtrType, p).getResult();
    mlir::Value elem = rewriter.create<emitc::SubscriptOp>(
        loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pc), idxLit(0)).getResult();
    mlir::Value v = rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, v).getResult();
  };
  // Load one uint8 byte at (base + off) ZERO-extended to uint32 (for the scales_h
  // little-endian assembly; a logical-shift domain -- ggml's uint16 read).
  auto loadU8Uint = [&](mlir::Value base, int64_t off) -> mlir::Value {
    mlir::Value p = rewriter.create<emitc::AddOp>(loc, inputPtrType, base, sizeLit(off)).getResult();
    mlir::Value pc = rewriter.create<emitc::CastOp>(loc, u8PtrType, p).getResult();
    mlir::Value elem = rewriter.create<emitc::SubscriptOp>(
        loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pc), idxLit(0)).getResult();
    mlir::Value v = rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
    return rewriter.create<emitc::CastOp>(loc, uintType, v).getResult();
  };
  auto fp16ReadAt = [&](mlir::Value xb, int64_t off) -> mlir::Value {
    mlir::Value addr = off == 0 ? xb
                                : rewriter.create<emitc::AddOp>(loc, inputPtrType, xb,
                                                                sizeLit(off)).getResult();
    return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                          mlir::ValueRange{addr}, opName, role,
                          llvm::StringRef("fcvt.s.h"));
  };
  // The nvfp4 per-sub-block UE4M3 fp8 -> fp32 HALF scale (ggml_ue4m3_to_fp32,
  // ggml-impl.h): e==0||e==0x7F -> 0; exp=(e>>3)&0xF, man=e&7; raw = exp==0 ?
  // ldexpf(man,-9) : ldexpf(1+man/8, exp-7); result = raw*0.5f. Structured emitc,
  // byte-identical to the scalar reference NVFP4 seam.
  auto ue4m3ScaleAt = [&](mlir::Value xb, int64_t off) -> mlir::Value {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "ue4m3_scale"));
    mlir::Value e32 = loadU8Uint(xb, off);
    mlir::Value expU = rewriter.create<emitc::BitwiseAndOp>(
        loc, uintType, rewriter.create<emitc::BitwiseRightShiftOp>(loc, uintType, e32, uLit(3)),
        uLit(0xF)).getResult();
    mlir::Value manU = rewriter.create<emitc::BitwiseAndOp>(loc, uintType, e32, uLit(0x7)).getResult();
    mlir::Value expInt = rewriter.create<emitc::CastOp>(loc, intType, expU).getResult();
    mlir::Value manFloat = i2f(rewriter.create<emitc::CastOp>(loc, intType, manU).getResult());
    mlir::Value denormRaw = rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{floatType}, "ldexpf",
        mlir::ValueRange{manFloat, intLit(-9)}).getResult(0);
    mlir::Value normMant = rewriter.create<emitc::AddOp>(
        loc, floatType, floatLit("1.0f"),
        rewriter.create<emitc::DivOp>(loc, floatType, manFloat, floatLit("8.0f"))).getResult();
    mlir::Value normExp = rewriter.create<emitc::SubOp>(loc, intType, expInt, intLit(7)).getResult();
    mlir::Value normRaw = rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{floatType}, "ldexpf",
        mlir::ValueRange{normMant, normExp}).getResult(0);
    mlir::Value isDenorm = rewriter.create<emitc::CmpOp>(
        loc, boolType, emitc::CmpPredicate::eq, expU, uLit(0)).getResult();
    mlir::Value raw = rewriter.create<emitc::ConditionalOp>(
        loc, floatType, isDenorm, denormRaw, normRaw).getResult();
    mlir::Value scaled = fMul(raw, floatLit("0.5f"));
    mlir::Value isZero = rewriter.create<emitc::CmpOp>(
        loc, boolType, emitc::CmpPredicate::eq, e32, uLit(0)).getResult();
    mlir::Value isSpec = rewriter.create<emitc::CmpOp>(
        loc, boolType, emitc::CmpPredicate::eq, e32, uLit(0x7F)).getResult();
    mlir::Value special = rewriter.create<emitc::LogicalOrOp>(
        loc, boolType, isZero, isSpec).getResult();
    return rewriter.create<emitc::ConditionalOp>(
        loc, floatType, special, floatLit("0.0f"), scaled).getResult();
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // ---- The 16-entry int8 codebook, emitted ONCE as a function-local static ----
  // mxfp4/nvfp4 share the FP4 e2m1 codebook; iq4_nl/iq4_xs share the non-linear
  // codebook (the SAME 16-entry anchors the block-dot vec_dot lowerings render).
  static const int kvaluesMxfp4[16] = {0, 1, 2, 3,  4,  6,  8,  12,
                                       0, -1, -2, -3, -4, -6, -8, -12};
  static const int kvaluesIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1,    13,   25,  38,  53,  69,  89,  113};
  // The 16-entry table + its name ride plan.codebookTable ([F-1]: the FP4-class
  // (mxfp4/nvfp4) vs non-linear (iq4_nl/iq4_xs) table is a STRUCTURAL plan fact, NOT the
  // format name).
  const bool isFp4Table = plan.codebookTable == ::weft::CodebookTable::Fp4E2M1;
  llvm::StringRef codebookName = isFp4Table ? "weft_dequant_mxfp4_kvalues"
                                            : "weft_dequant_iq4nl_kvalues";
  {
    llvm::ArrayRef<int> entries = isFp4Table ? llvm::ArrayRef<int>(kvaluesMxfp4)
                                             : llvm::ArrayRef<int>(kvaluesIq4nl);
    std::string decl = "static const int8_t " + codebookName.str() + "[" +
                       std::to_string(plan.codebookEntries) + "] = {";
    for (size_t i = 0; i < entries.size(); ++i) {
      if (i) decl += ", ";
      decl += std::to_string(entries[i]);
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);
  }

  // vint8<load>_t values = __riscv_vle8_v_i8<load>(<codebook>, codebookEntries);
  // (broadcast the codebook into ONE vreg, reused by every vrgather -- register-resident,
  // NO memory gather). The load LMUL + entry count ride the PLAN.
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "codebook_table_load"));
  std::string tableLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
  mlir::Value values = emitOpaqueCallBuilt(
      rewriter, loc, i8VecType, tableLoadCallee, opName, role,
      [&](mlir::OpBuilder &, mlir::Location) -> llvm::SmallVector<mlir::Value> {
        mlir::Value tbl = rewriter.create<emitc::LiteralOp>(loc, i8PtrType, codebookName);
        return {tbl, sizeLit(plan.codebookEntries)};
      });

  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb = rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1), nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter.create<emitc::AddOp>(loc, inputPtrType, input, mulSz(ib, sizeLit(stride))).getResult();
    mlir::Value ybRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, output, mulSz(ib, sizeLit(qk))).getResult();
    mlir::Value yb = rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    // Emit ONE half-group codebook pipeline over `nLanes` packed nibble bytes at
    // (xb + qsByteOff): low/high nibble split -> vrgather -> sext -> fcvt -> vfmul by
    // `scale` -> vse32 to (yb + outLo) and (yb + outHi).
    auto emitHalfGroup = [&](int64_t qsByteOff, int64_t nLanes, mlir::Value scale,
                             int64_t outLo, int64_t outHi) {
      mlir::Value qsPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsByteOff)).getResult();
      mlir::Value qsU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qsPtr).getResult();
      mlir::Value w = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, coreLmul, "u8"),
                                     mlir::ValueRange{qsU8, sizeLit(nLanes)}, opName, role);
      mlir::Value idxLo = emitOpaqueCall(rewriter, loc, u8VecType, ("__riscv_vand_vx_u8" + coreLmul).str(),
                                         mlir::ValueRange{w, intLit(0x0F), sizeLit(nLanes)}, opName, role);
      mlir::Value idxHi = emitOpaqueCall(rewriter, loc, u8VecType, ("__riscv_vsrl_vx_u8" + coreLmul).str(),
                                         mlir::ValueRange{w, intLit(4), sizeLit(nLanes)}, opName, role);
      auto lane = [&](mlir::Value idx, int64_t outOff) {
        mlir::Value g = emitOpaqueCall(rewriter, loc, i8VecType, ("__riscv_vrgather_vv_i8" + coreLmul).str(),
                                       mlir::ValueRange{values, idx, sizeLit(nLanes)}, opName, role);
        mlir::Value w32 = emitOpaqueCall(rewriter, loc, i32VecType, ("__riscv_vsext_vf4_i32" + wideLmul).str(),
                                         mlir::ValueRange{g, sizeLit(nLanes)}, opName, role);
        mlir::Value f = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, wideLmul, "f32"),
                                       mlir::ValueRange{w32, sizeLit(nLanes)}, opName, role);
        mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, wideLmul, "f32"),
                                       mlir::ValueRange{f, scale, sizeLit(nLanes)}, opName, role);
        mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outOff)).getResult();
        emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, wideLmul, "f32"),
                           mlir::ValueRange{yStore, r, sizeLit(nLanes)}, opName, role);
      };
      lane(idxLo, outLo);
      lane(idxHi, outHi);
    };

    if (isMx || isNl) {
      // qk=32 nibble codebook: y[j] = d*kv[qs[j]&0xF]; y[j+16] = d*kv[qs[j]>>4].
      // mxfp4: qs @+1, E8M0 block scale; iq4_nl: qs @+2, fp16 d block scale.
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, isMx ? "mxfp4_decode" : "iq4_nl_decode"));
      mlir::Value d = isMx ? emitE8M0HalfScale(rewriter, loc, xb, opName, role)
                           : fp16ReadAt(xb, 0);
      // qs base + strip lanes ride the PLAN (qsBase = 1 mxfp4 / 2 iq4_nl; strip = qk/2).
      emitHalfGroup(/*qsByteOff=*/qsBase, /*nLanes=*/stripLanes, d, /*outLo=*/0, /*outHi=*/16);
    } else if (isNv) {
      // Four 16-element UE4M3-scaled sub-blocks; qs @+4, 8 bytes/sub. Per sub s:
      // y[s*16 + j] = d[s]*kv[qs[s*8+j]&0xF]; y[s*16+8 + j] = d[s]*kv[qs[s*8+j]>>4].
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "nvfp4_sub_decode"));
      for (int64_t s = 0; s < 4; ++s) {
        mlir::Value d = ue4m3ScaleAt(xb, s);
        // qs base (4) + per-sub stride (8) off the PLAN; strip = sub-block half (8).
        emitHalfGroup(/*qsByteOff=*/qsBase + s * 8, /*nLanes=*/stripLanes, d, /*outLo=*/s * 16, /*outHi=*/s * 16 + 8);
      }
    } else if (isXs) { // iq4_xs
      // d@0, scales_h(u16)@2, scales_l[4]@4, qs[128]@8. Per ib (8 sub-blocks of 32):
      // ls = (scales_l[ib/2] >> 4*(ib%2))&0xF | ((scales_h >> 2*ib)&3)<<4; dl=d*(ls-32);
      // y[ib*32 + j] = dl*kv[qs[ib*16+j]&0xF]; y[ib*32+16 + j] = dl*kv[qs[ib*16+j]>>4].
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "iq4_xs_decode"));
      mlir::Value d = fp16ReadAt(xb, 0);
      // scales_h assembled little-endian from bytes @2,@3 into int (positive, <=0xFFFF).
      mlir::Value sh = iOr(loadU8Int(xb, 2), iShl(loadU8Int(xb, 3), intLit(8)));
      for (int64_t ib32 = 0; ib32 < 8; ++ib32) {
        mlir::Value scl = loadU8Int(xb, 4 + ib32 / 2);
        mlir::Value low = iAnd(iShr(scl, intLit(4 * (ib32 % 2))), intLit(0xF));
        mlir::Value hi = iShl(iAnd(iShr(sh, intLit(2 * ib32)), intLit(3)), intLit(4));
        mlir::Value dl = fMul(d, i2f(iSub(iOr(low, hi), intLit(32))));
        // qs base (8) + per-sub stride (16) off the PLAN; strip = sub-block half (16).
        emitHalfGroup(/*qsByteOff=*/qsBase + ib32 * 16, /*nLanes=*/stripLanes, dl,
                      /*outLo=*/ib32 * 32, /*outHi=*/ib32 * 32 + 16);
      }
    } else {
      // Unreachable for the four codebook scale models (one of isMx/isNl/isNv/isXs is
      // always true here); fail-closed on any future unhandled CodebookScaleModel rather
      // than emitting an empty body.
      return rewriter.notifyMatchFailure(
          loc, "codebook gather plan carries an unhandled scale model");
    }
  }
  return mlir::success();
}

// ============================================================================
// The OWNED REAL-VECTOR ternary super-block dequantize_row body (B线批3 ternary
// de-lottery · [L-8] · ISSUE-001 reverse · closes the ISSUE-002 codegen-lottery for the
// tq1_0/tq2_0 ternary super-blocks). Unlike the tiny-codebook fan-out there is NO
// codebook table and NO gather: the ternary {-1,0,1} value is decoded by PURE ARITHMETIC.
//   tq2_0 (2-bit): q = (qs >> (2l)) & 3 (vsrl_vx / vand_vx, u8m1), reinterpret to i8,
//                  vsext_vf4 -> i32m4, subtract 1, int->float (vfcvt), scaled by the
//                  fp16 d in ONE vfmul, stored (vse32).
//   tq1_0 (base-3): q = (uint8_t)(byte * pow3[n]) (vmul_vx u8m1, mod-256), xi =
//                  ((uint16_t)q * 3) >> 8 (vzext_vf2 -> u16m2, vmul_vx, vsrl_vx),
//                  reinterpret to i16, vsext_vf2 -> i32m4, subtract 1, vfcvt, ONE vfmul
//                  by d, vse32. qs packs 5 base-3 digits/byte (n=0..4), qh packs 4
//                  (n=0..3). The `* pow3[n]` base-3 digit extraction is the OWNED vector
//                  analogue of the block-dot vwmulu.vx powers-of-3 unpack, but STREAMING
//                  (no reduction). The integer ternary decode mirrors the scalar
//                  the retired scalar reference byte-for-byte; the ONE vfmul by d ==
//                  ggml's single `(q-1)*d` / `(xi-1)*d` mul -> no fp-contraction
//                  ambiguity. Byte-exact to dequantize_row_{tq1_0,tq2_0} by construction.
// All lane groups are emitted with nLanes<=16 so every widened LMUL (u16m2 / i32m4 /
// f32m4) fits VLMAX at VLEN128 (and processes exactly nLanes elements on any VLEN>=128).
// Streaming sibling of emitDequantizeRowCodebookVectorBody (no accumulator, no gather).
// ============================================================================
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowTernaryVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    bool isTq1) const {
  // Per-format ternary super-block geometry (byte-exact ggml block_tqX AoS facts, NOT
  // knobs): tq2_0 { qs[64]; d(fp16); } stride=66; tq1_0 { qs[48]; qh[4]; d(fp16); }
  // stride=54. Both QK_K=256.
  const int64_t qk = 256;
  const int64_t stride = isTq1 ? 54 : 66;
  const int64_t dOff = isTq1 ? 52 : 64;

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
  mlir::Type i8VecType = emitc::OpaqueType::get(ctx, "vint8m1_t");
  mlir::Type u16VecType = emitc::OpaqueType::get(ctx, "vuint16m2_t");
  mlir::Type i16VecType = emitc::OpaqueType::get(ctx, "vint16m2_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m4_t");
  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
  auto mulSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::MulOp>(loc, sizeType, a, b).getResult();
  };
  auto fp16ReadAt = [&](mlir::Value xb, int64_t off) -> mlir::Value {
    mlir::Value addr = off == 0 ? xb
                                : rewriter.create<emitc::AddOp>(loc, inputPtrType, xb,
                                                                sizeLit(off)).getResult();
    return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                          mlir::ValueRange{addr}, opName, role,
                          llvm::StringRef("fcvt.s.h"));
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb = rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1), nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb = rewriter.create<emitc::AddOp>(loc, inputPtrType, input, mulSz(ib, sizeLit(stride))).getResult();
    mlir::Value ybRaw = rewriter.create<emitc::AddOp>(loc, outputPtrType, output, mulSz(ib, sizeLit(qk))).getResult();
    mlir::Value yb = rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d_scale"));
    mlir::Value d = fp16ReadAt(xb, dOff);

    // Load `nLanes` packed bytes at (xb + srcOff) as u8m1.
    auto loadBytes = [&](int64_t srcOff, int64_t nLanes) -> mlir::Value {
      mlir::Value qsPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(srcOff)).getResult();
      mlir::Value qsU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qsPtr).getResult();
      return emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                            mlir::ValueRange{qsU8, sizeLit(nLanes)}, opName, role);
    };
    // Common tail: `w32` holds the RAW ternary digit {0,1,2} sign-extended to i32m4;
    // subtract 1 -> {-1,0,1}, int->float, scale by d in ONE vfmul, store to (yb+outOff).
    auto emitScaleStore = [&](mlir::Value w32, int64_t nLanes, int64_t outOff) {
      mlir::Value sub1 = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsub_vx_i32m4",
                                        mlir::ValueRange{w32, intLit(1), sizeLit(nLanes)}, opName, role);
      mlir::Value f = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                     mlir::ValueRange{sub1, sizeLit(nLanes)}, opName, role);
      mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, "m4", "f32"),
                                     mlir::ValueRange{f, d, sizeLit(nLanes)}, opName, role);
      mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outOff)).getResult();
      emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                         mlir::ValueRange{yStore, r, sizeLit(nLanes)}, opName, role);
    };

    if (!isTq1) {
      // tq2_0: over qs[64], each byte packs FOUR 2-bit lanes. For j in {0,32}, l in
      // 0..3, m in 0..31: q = (qs[j+m] >> (2l)) & 3; y[(j/32)*128 + l*32 + m] = (q-1)*d.
      // Each 32-lane (jg,l) group is split into two 16-lane halves (VLEN128 VLMAX).
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "tq2_0_decode"));
      for (int64_t jg = 0; jg < 2; ++jg) {
        for (int64_t l = 0; l < 4; ++l) {
          for (int64_t half = 0; half < 2; ++half) {
            int64_t srcOff = jg * 32 + half * 16;
            int64_t outOff = jg * 128 + l * 32 + half * 16;
            mlir::Value w = loadBytes(srcOff, 16);
            mlir::Value sh =
                l == 0 ? w
                       : emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                        mlir::ValueRange{w, intLit(2 * l), sizeLit(16)}, opName, role);
            mlir::Value masked = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                                mlir::ValueRange{sh, intLit(3), sizeLit(16)}, opName, role);
            mlir::Value qi8 = emitOpaqueCall(rewriter, loc, i8VecType, "__riscv_vreinterpret_v_u8m1_i8m1",
                                             mlir::ValueRange{masked}, opName, role);
            mlir::Value w32 = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsext_vf4_i32m4",
                                             mlir::ValueRange{qi8, sizeLit(16)}, opName, role);
            emitScaleStore(w32, 16, outOff);
          }
        }
      }
    } else {
      // tq1_0: base-3 packed. pow3 = {1,3,9,27,81}; q = (uint8_t)(byte * pow3[n]);
      // xi = ((uint16_t)q * 3) >> 8; y[out] = (xi-1)*d.
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "tq1_0_decode"));
      static const int64_t pow3[5] = {1, 3, 9, 27, 81};
      // The base-3 digit-n extraction for `nLanes` bytes at srcOff, storing to outOff.
      auto emitTq1Chunk = [&](int64_t srcOff, int64_t n, int64_t nLanes, int64_t outOff) {
        mlir::Value w = loadBytes(srcOff, nLanes);
        mlir::Value q = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vmul_vx_u8m1",
                                       mlir::ValueRange{w, intLit(pow3[n]), sizeLit(nLanes)}, opName, role);
        mlir::Value q16 = emitOpaqueCall(rewriter, loc, u16VecType, "__riscv_vzext_vf2_u16m2",
                                         mlir::ValueRange{q, sizeLit(nLanes)}, opName, role);
        mlir::Value t = emitOpaqueCall(rewriter, loc, u16VecType, "__riscv_vmul_vx_u16m2",
                                       mlir::ValueRange{q16, intLit(3), sizeLit(nLanes)}, opName, role);
        mlir::Value xi = emitOpaqueCall(rewriter, loc, u16VecType, "__riscv_vsrl_vx_u16m2",
                                        mlir::ValueRange{t, intLit(8), sizeLit(nLanes)}, opName, role);
        mlir::Value xii = emitOpaqueCall(rewriter, loc, i16VecType, "__riscv_vreinterpret_v_u16m2_i16m2",
                                         mlir::ValueRange{xi}, opName, role);
        mlir::Value w32 = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsext_vf2_i32m4",
                                         mlir::ValueRange{xii, sizeLit(nLanes)}, opName, role);
        emitScaleStore(w32, nLanes, outOff);
      };
      // main qs (j=0): out[n*32 + m], n=0..4, m=0..31 -> two 16-lane halves.
      for (int64_t n = 0; n < 5; ++n)
        for (int64_t half = 0; half < 2; ++half)
          emitTq1Chunk(/*srcOff=*/half * 16, n, /*nLanes=*/16, /*outOff=*/n * 32 + half * 16);
      // tail qs (j=32): out[160 + n*16 + m], n=0..4, m=0..15.
      for (int64_t n = 0; n < 5; ++n)
        emitTq1Chunk(/*srcOff=*/32, n, /*nLanes=*/16, /*outOff=*/160 + n * 16);
      // qh (offset 48): out[240 + n*4 + j], n=0..3, j=0..3.
      for (int64_t n = 0; n < 4; ++n)
        emitTq1Chunk(/*srcOff=*/48, n, /*nLanes=*/4, /*outOff=*/240 + n * 4);
    }
  }
  return mlir::success();
}

// Lower the CONSTRUCTED streaming dequantize_row region: walk the
// weft_rvv.typed_dequantize_row_loop_body, extract its per-block DECODE brick
// (weft_rvv.dequantize_row_decode_core) + the VOID yield, enforce the anti-bypass
// invariant (the brick's block_index MUST be the region induction variable / region
// arg 0, so the ABI bases are sourced from the BRICK not inferred), and re-emit the
// whole nb block loop + per-block decode via the SHARED body emitter -- byte-exact to
// the dispatch-wired q8_0 monolith. The whole loop is emitter-inlined by the brick
// lowering (the streaming analog of q1_0/nvfp4's flat single-core-brick emit).
mlir::LogicalResult VariantToEmitCFunc::emitTypedDequantizeRowLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  weftrvv::TypedDequantizeRowLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<weftrvv::TypedDequantizeRowLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed dequantize_row loop body missing the op");

  weftrvv::DequantizeRowDecodeCoreOp coreOp;
  weftrvv::TypedDequantizeRowLoopYieldOp yieldOp;
  loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
    if (auto o = llvm::dyn_cast<weftrvv::DequantizeRowDecodeCoreOp>(bodyOp))
      coreOp = o;
    else if (auto o =
                 llvm::dyn_cast<weftrvv::TypedDequantizeRowLoopYieldOp>(bodyOp))
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
  mlir::StringAttr mechanismAttr = coreOp.getDequantMechanismAttr();
  if (!mechanismAttr)
    return rewriter.notifyMatchFailure(
        loopBody, "typed dequantize_row emission requires construction-owned "
                  "dequant_mechanism and a complete final formula plan");
  llvm::StringRef mechanism = mechanismAttr.getValue();
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

  llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
  // The OPTIONAL g-axis grid geometry descriptor (codebook grid ENTRY byte-width),
  // stamped by the dequant-stream front door ONLY for the three owned grid-codebook
  // decode leaves (iq3_s / iq2_xs / iq1_m). The owned narrow-per-entry grid dequant
  // bodies READ this entry lane count from the descriptor instead of baking the format
  // constant into the mechanism body (律2); the shared IQ-grid / codebook-grid
  // dispatchers fail closed (no value_or self-supply) if a consuming leaf reaches emit
  // without it. Absent for every flat / K-quant / non-grid leaf (they never read it).
  std::optional<int64_t> codebookEntryLanes;
  if (mlir::IntegerAttr entryLanesAttr =
          coreOp->getAttrOfType<mlir::IntegerAttr>("codebook_entry_lanes"))
    codebookEntryLanes = entryLanesAttr.getInt();
  mlir::StringAttr carrierAttr = coreOp.getCarrierKindAttr();
  if (mechanism == "nibble-decode") {
    mlir::StringAttr loadLMUL =
        coreOp->getAttrOfType<mlir::StringAttr>("dequant_load_lmul");
    mlir::IntegerAttr stripLanes =
        coreOp->getAttrOfType<mlir::IntegerAttr>("dequant_strip_lanes");
    if (!carrierAttr || carrierAttr.getValue() != "nibble4" || !loadLMUL ||
        !stripLanes)
      return rewriter.notifyMatchFailure(
          loopBody, "nibble emitter requires the complete final formula plan");
    ::weft::NibbleDecodePlan plan{};
    plan.mechanism = ::weft::DequantMechanism::NibbleDecode;
    plan.carrier = ::weft::NibbleCarrier::Nibble4;
    plan.weightBlockStride = coreOp.getWeightBlockStrideAttr().getInt();
    plan.scaleByteOffset = coreOp.getScaleByteOffsetAttr().getInt();
    plan.quantByteOffset = coreOp.getQuantByteOffsetAttr().getInt();
    plan.nibbleBias = coreOp.getNibbleBiasAttr().getInt();
    plan.hasMin = static_cast<bool>(coreOp.getMinByteOffsetAttr());
    plan.minByteOffset =
        plan.hasMin ? coreOp.getMinByteOffsetAttr().getInt() : 0;
    plan.hasQh = static_cast<bool>(coreOp.getQhByteOffsetAttr());
    plan.qhByteOffset =
        plan.hasQh ? coreOp.getQhByteOffsetAttr().getInt() : 0;
    plan.loadLMUL = loadLMUL.getValue();
    plan.stripLanes = stripLanes.getInt();
    if (plan.loadLMUL != "m1" || plan.stripLanes <= 0)
      return rewriter.notifyMatchFailure(
          loopBody, "nibble selected plan has no mechanical realization");
    return emitDequantizeRowNibbleVectorBody(rewriter, loc, weightBase, output,
                                             avlArg, sizeType, opName, role,
                                             plan);
  }
  if (mechanism == "codebook-gather") {
    mlir::StringAttr scaleModelAttr = coreOp.getCodebookScaleModelAttr();
    mlir::StringAttr tableAttr = coreOp.getCodebookGatherTableAttr();
    mlir::IntegerAttr entriesAttr = coreOp.getCodebookGatherEntriesAttr();
    mlir::IntegerAttr stripAttr =
        coreOp->getAttrOfType<mlir::IntegerAttr>("dequant_strip_lanes");
    mlir::StringAttr loadLMULAttr =
        coreOp->getAttrOfType<mlir::StringAttr>("dequant_load_lmul");
    if (!scaleModelAttr || !tableAttr || !entriesAttr || !stripAttr ||
        !loadLMULAttr)
      return rewriter.notifyMatchFailure(
          loopBody,
          "codebook gather emitter requires the complete final formula plan");

    std::optional<::weft::CodebookScaleModel> scaleModel =
        ::weft::parseCodebookScaleModel(scaleModelAttr.getValue());
    std::optional<::weft::CodebookTable> table =
        ::weft::parseCodebookTable(tableAttr.getValue());
    if (!scaleModel || !table)
      return rewriter.notifyMatchFailure(
          loopBody,
          "codebook gather final plan carries an unknown scale model/table");

    ::weft::CodebookGatherPlan plan{};
    plan.mechanism = ::weft::DequantMechanism::CodebookGather;
    plan.scaleModel = *scaleModel;
    plan.codebookTable = *table;
    plan.codebookEntries = entriesAttr.getInt();
    plan.codebookByteOffset = coreOp.getQuantByteOffsetAttr().getInt();
    plan.superBlockElements = coreOp.getQkAttr().getInt();
    plan.weightBlockStride = coreOp.getWeightBlockStrideAttr().getInt();
    plan.loadLMUL = loadLMULAttr.getValue();
    plan.stripLanes = stripAttr.getInt();
    plan.legality.isLegal = true;
    if (plan.codebookEntries != 16 || plan.stripLanes <= 0 ||
        (plan.loadLMUL != "mf2" && plan.loadLMUL != "m1" &&
         plan.loadLMUL != "m2"))
      return rewriter.notifyMatchFailure(
          loopBody, "codebook gather final plan has no mechanical realization");
    return emitDequantizeRowCodebookVectorBody(rewriter, loc, weightBase, output,
                                               avlArg, sizeType, opName, role,
                                               plan);
  }
  // Phase-4 (DequantMechanismPlan family #3): the QK_K=256 K-quant super-block family
  // (q2_K/q3_K/q4_K/q5_K/q6_K) decode is assembled into a KQuantScaleMin MechanismPlan
  // (weft::KQuantScaleMinPlan) by constructKQuantScaleMinPlan from the stamped decode_core
  // geometry facts + the per-format scale model, and the K-quant emitters READ plan.* --
  // NOT the format name (the per-format qk/stride/scale-block/quant/sub-scale/min/high-bit
  // re-derivation is RETIRED into the plan). The dispatch tests plan.mechanism ==
  // KQuantScaleMin ([F-1]: name AS DATA, not an execution key); format survives only as
  // plan.provenanceFormat. Byte-exact reproduce-current: loadLMUL/stripLanes are pinned to
  // the FIXED ggml-ABI super-block geometry (phase-3-kquant c-drives them f(VLEN)). [K-10]:
  // KQuantScaleMinPlan is the KQuantScaleMin mechanism's OWN plan (the 3rd of 5); the five
  // formats parametrize the (already-separate) bit-unpack + scale/min-fold bodies via
  // plan.scaleModel -- a PARAMETRIC leaf shape, NOT a plan-internal mechanism switch. The
  // scale model is derived once here from the block-type identity (the SAME
  // format->ABI-facts mapping the construction table performs; a front-door scale-model
  // stamp is the phase-1-equivalent follow-up). The CONSTRUCTED path lowers to the OWNED
  // real-vector bodies (per-super-sub vle8 + vand/vsrl bit unpack + [q5_K/q6_K high-bit
  // merge] + vzext + vfcvt + fused vfmsac_vf (min formats) / vfmul_vf (single-mul) + vse32,
  // NO gather); the dispatch-wired monolith fallback keeps the scalar K-quant decode
  // reference -- the q8_0/nibble/codebook precedent.
  if (mechanism == "kquant-scale-min") {
    mlir::StringAttr modelAttr =
        coreOp->getAttrOfType<mlir::StringAttr>("kquant_scale_model");
    std::optional<::weft::KQuantScaleModel> model =
        modelAttr ? ::weft::parseKQuantScaleModel(modelAttr.getValue())
                  : std::nullopt;
    mlir::StringAttr loadLMUL =
        coreOp->getAttrOfType<mlir::StringAttr>("dequant_load_lmul");
    mlir::IntegerAttr stripLanes =
        coreOp->getAttrOfType<mlir::IntegerAttr>("dequant_strip_lanes");
    mlir::BoolAttr hasMin =
        coreOp->getAttrOfType<mlir::BoolAttr>("kquant_has_min");
    mlir::IntegerAttr minOffset =
        coreOp->getAttrOfType<mlir::IntegerAttr>("kquant_min_byte_offset");
    mlir::IntegerAttr subScaleOffset = coreOp->getAttrOfType<mlir::IntegerAttr>(
        "kquant_sub_scale_byte_offset");
    mlir::BoolAttr hasHighBit = coreOp->getAttrOfType<mlir::BoolAttr>(
        "kquant_has_high_bit_plane");
    mlir::IntegerAttr highBitOffset = coreOp->getAttrOfType<mlir::IntegerAttr>(
        "kquant_high_bit_byte_offset");
    if (!model || !loadLMUL || !stripLanes || !hasMin || !minOffset ||
        !subScaleOffset || !hasHighBit || !highBitOffset)
      return rewriter.notifyMatchFailure(
          loopBody, "kquant emitter requires the complete final formula plan");
    ::weft::KQuantScaleMinPlan plan{};
    plan.mechanism = ::weft::DequantMechanism::KQuantScaleMin;
    plan.scaleModel = *model;
    plan.superBlockElements = coreOp.getQkAttr().getInt();
    plan.weightBlockStride = coreOp.getWeightBlockStrideAttr().getInt();
    plan.quantByteOffset = coreOp.getQuantByteOffsetAttr().getInt();
    plan.scaleBlockByteOffset = coreOp.getScaleByteOffsetAttr().getInt();
    plan.hasMin = hasMin.getValue();
    plan.minByteOffset = minOffset.getInt();
    plan.subScaleByteOffset = subScaleOffset.getInt();
    plan.hasHighBitPlane = hasHighBit.getValue();
    plan.highBitByteOffset = highBitOffset.getInt();
    plan.loadLMUL = loadLMUL.getValue();
    plan.stripLanes = stripLanes.getInt();
    plan.legality.isLegal = true;
    if (plan.stripLanes <= 0 ||
        (plan.loadLMUL != "m1" && plan.loadLMUL != "m2"))
      return rewriter.notifyMatchFailure(
          loopBody, "kquant selected plan has no mechanical realization");
    switch (plan.scaleModel) {
    case ::weft::KQuantScaleModel::Q4K:
    case ::weft::KQuantScaleModel::Q5K:
      return emitDequantizeRowQ45KVectorBody(rewriter, loc, weightBase, output,
                                             avlArg, sizeType, opName, role, plan);
    case ::weft::KQuantScaleModel::Q2K:
      return emitDequantizeRowQ2KVectorBody(rewriter, loc, weightBase, output,
                                            avlArg, sizeType, opName, role, plan);
    case ::weft::KQuantScaleModel::Q3K:
      return emitDequantizeRowQ3KVectorBody(rewriter, loc, weightBase, output,
                                            avlArg, sizeType, opName, role, plan);
    case ::weft::KQuantScaleModel::Q6K:
      return emitDequantizeRowQ6KVectorBody(rewriter, loc, weightBase, output,
                                            avlArg, sizeType, opName, role, plan);
    }
  }
  // Phase-4 (DequantMechanismPlan family #4): the QK_K=256 IQ grid-table family
  // (iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s) decode is assembled into a GridLookup MechanismPlan
  // (weft::GridLookupPlan) by constructGridLookupPlan. The grid emitter reads plan.* --
  // the dispatch tests plan.mechanism == GridLookup
  // ([F-1]: name AS DATA, not an execution key), the per-format owned body is selected by
  // plan.leaf, and the grid ENTRY g-axis geometry rides plan.entryLanes; format survives only
  // as plan.provenanceFormat. Byte-exact reproduce-current. [K-10]: GridLookupPlan is the
  // GridLookup mechanism's OWN plan (the 4th of 5); the ternary siblings are a DIFFERENT
  // mechanism (TernaryDecode) with their own plan below -- NOT folded here.
  if (mechanism == "grid-lookup") {
    mlir::StringAttr leafAttr =
        coreOp->getAttrOfType<mlir::StringAttr>("grid_decode_leaf");
    std::optional<::weft::GridDecodeLeaf> leaf =
        leafAttr ? ::weft::parseGridDecodeLeaf(leafAttr.getValue())
                 : std::nullopt;
    if (!leaf)
      return rewriter.notifyMatchFailure(
          loopBody, "grid emitter requires a recognized typed grid leaf");
    ::weft::GridLookupPlan plan{};
    plan.mechanism = ::weft::DequantMechanism::GridLookup;
    plan.leaf = *leaf;
    plan.hasEntryLanes = codebookEntryLanes.has_value();
    plan.entryLanes = codebookEntryLanes.value_or(0);
    plan.legality.isLegal = true;
    return emitDequantizeRowIQGridBodyShared(rewriter, loc, weightBase, output,
                                             avlArg, sizeType, opName, role, plan);
  }
  // Phase-4 (DequantMechanismPlan family #5, the [K-10] ternary split): the ternary family
  // (iq1_s/iq1_m ternary iq1s_grid + the tq1_0/tq2_0 base-3 / 2-bit ternary super-blocks)
  // decode is assembled into a TernaryDecode MechanismPlan (weft::TernaryDecodePlan) by the
  // constructTernaryDecodePlan. Ternary is a SEPARATE mechanism -- it does
  // NOT consult the grid registry (grid != ternary RESTORED for the dequant-row head; the
  // GridDecodePlan registry's own iq1_s/iq1_m lumping serves the block-dot head, untouched).
  // The ternary emitter READS plan.* -- the dispatch tests plan.mechanism == TernaryDecode,
  // the per-format owned body is selected by plan.leaf, and the iq1 grid ENTRY g-axis rides
  // plan.entryLanes. Byte-exact reproduce-current. (The iq4_nl/iq4_xs/mxfp4/nvfp4 codebook
  // leaves are yet another mechanism -- CodebookGather -- dispatched above.)
  if (mechanism == "ternary-decode") {
    mlir::StringAttr leafAttr =
        coreOp->getAttrOfType<mlir::StringAttr>("ternary_decode_leaf");
    std::optional<::weft::TernaryDecodeLeaf> leaf =
        leafAttr ? ::weft::parseTernaryDecodeLeaf(leafAttr.getValue())
                 : std::nullopt;
    if (!leaf)
      return rewriter.notifyMatchFailure(
          loopBody, "ternary emitter requires a recognized typed ternary leaf");
    ::weft::TernaryDecodePlan plan{};
    plan.mechanism = ::weft::DequantMechanism::TernaryDecode;
    plan.leaf = *leaf;
    plan.hasEntryLanes = codebookEntryLanes.has_value();
    plan.entryLanes = codebookEntryLanes.value_or(0);
    plan.legality.isLegal = true;
    return emitDequantizeRowTernaryDecodeBodyShared(rewriter, loc, weightBase,
                                                    output, avlArg, sizeType,
                                                    opName, role, plan);
  }
  if (mechanism == "binary-sign")
    return emitDequantizeRowBinarySignBodyShared(
        rewriter, loc, weightBase, output, avlArg, sizeType, opName, role,
        coreOp.getQkAttr().getInt(),
        coreOp.getWeightBlockStrideAttr().getInt(),
        coreOp.getScaleByteOffsetAttr().getInt(),
        coreOp.getQuantByteOffsetAttr().getInt());
  if (mechanism == "int8-scale" && carrierAttr &&
      carrierAttr.getValue() == "bare_int8")
    return emitDequantizeRowQ8_0VectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role,
                                           coreOp.getQkAttr().getInt(),
                                           coreOp.getWeightBlockStrideAttr().getInt(),
                                           coreOp.getScaleByteOffsetAttr().getInt(),
                                           coreOp.getQuantByteOffsetAttr().getInt());
  return rewriter.notifyMatchFailure(
      loopBody, "selected dequant mechanism has no typed emission realization");
}

// Mechanically emit the typed BinarySign plan. Format provenance has no dispatch
// role at this boundary.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowBinarySignBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    int64_t qk, int64_t weightBlockStride, int64_t scaleByteOffset,
    int64_t quantByteOffset) const {
  if (qk <= 0 || qk % 8 != 0 || weightBlockStride <= 0 ||
      scaleByteOffset < 0 || quantByteOffset < 0)
    return rewriter.notifyMatchFailure(
        loc, "binary-sign emitter received an invalid typed plan");

  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type boolType = rewriter.getI1Type();
  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type inputPtrType = input.getType();
  mlir::Type outputPtrType = output.getType();

  auto sizeLit = [&](int64_t value) {
    return emitSizeLit(rewriter, loc, sizeType, value);
  };
  auto intLit = [&](int64_t value) {
    return emitSizeLit(rewriter, loc, intType, value);
  };
  auto idxLit = [&](int64_t value) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, indexType,
                                              std::to_string(value));
  };
  auto mulSize = [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
    return rewriter.create<emitc::MulOp>(loc, sizeType, lhs, rhs).getResult();
  };
  auto shiftRight = [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
    return rewriter
        .create<emitc::BitwiseRightShiftOp>(loc, intType, lhs, rhs)
        .getResult();
  };
  auto bitAnd = [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
    return rewriter.create<emitc::BitwiseAndOp>(loc, intType, lhs, rhs)
        .getResult();
  };
  auto loadByteAsInt = [&](mlir::Value base,
                           mlir::Value byteOffset) -> mlir::Value {
    mlir::Value pointer =
        rewriter.create<emitc::AddOp>(loc, inputPtrType, base, byteOffset)
            .getResult();
    mlir::Value element =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pointer),
                idxLit(0))
            .getResult();
    mlir::Value byte =
        rewriter.create<emitc::LoadOp>(loc, constU8Type, element).getResult();
    return rewriter.create<emitc::CastOp>(loc, intType, byte).getResult();
  };
  auto storeFloat = [&](mlir::Value base, int64_t elementOffset,
                        mlir::Value value) {
    mlir::Value pointer =
        rewriter
            .create<emitc::AddOp>(loc, floatPtrType, base,
                                  sizeLit(elementOffset))
            .getResult();
    mlir::Value element =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(pointer),
                idxLit(0))
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, element, value);
  };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "super_block_count"));
  mlir::Value blockCount =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));
  auto blockLoop = rewriter.create<emitc::ForOp>(
      loc, sizeLit(0), blockCount, sizeLit(1), /*bodyBuilder=*/nullptr);
  mlir::Value blockIndex = blockLoop.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());

    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value inputBlock =
        rewriter
            .create<emitc::AddOp>(
                loc, inputPtrType, input,
                mulSize(blockIndex, sizeLit(weightBlockStride)))
            .getResult();
    mlir::Value outputBlockRaw =
        rewriter
            .create<emitc::AddOp>(loc, outputPtrType, output,
                                  mulSize(blockIndex, sizeLit(qk)))
            .getResult();
    mlir::Value outputBlock =
        rewriter.create<emitc::CastOp>(loc, floatPtrType, outputBlockRaw)
            .getResult();

    mlir::Value scaleAddress =
        scaleByteOffset == 0
            ? inputBlock
            : rewriter
                  .create<emitc::AddOp>(loc, inputPtrType, inputBlock,
                                        sizeLit(scaleByteOffset))
                  .getResult();
    mlir::Value scale = emitOpaqueCall(
        rewriter, loc, floatType, "(float)*(const _Float16 *)",
        mlir::ValueRange{scaleAddress}, opName, role,
        llvm::StringRef("fcvt.s.h"));
    mlir::Value negativeScale =
        rewriter.create<emitc::UnaryMinusOp>(loc, floatType, scale).getResult();
    mlir::Value zero = intLit(0);

    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "binary_sign_decode"));
    for (int64_t byteIndex = 0; byteIndex < qk / 8; ++byteIndex) {
      mlir::Value packed = loadByteAsInt(
          inputBlock, sizeLit(quantByteOffset + byteIndex));
      for (int64_t bitIndex = 0; bitIndex < 8; ++bitIndex) {
        mlir::Value bit =
            bitAnd(shiftRight(packed, intLit(bitIndex)), intLit(1));
        mlir::Value isSet =
            rewriter
                .create<emitc::CmpOp>(loc, boolType,
                                      emitc::CmpPredicate::ne, bit, zero)
                .getResult();
        mlir::Value decoded =
            rewriter
                .create<emitc::ConditionalOp>(loc, floatType, isSet, scale,
                                              negativeScale)
                .getResult();
        storeFloat(outputBlock, byteIndex * 8 + bitIndex, decoded);
      }
    }
  }
  return mlir::success();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
