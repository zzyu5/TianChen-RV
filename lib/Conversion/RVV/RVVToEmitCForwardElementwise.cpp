#include "RVVToEmitCInternal.h"
#include "Weft/Conversion/RVV/RVVToEmitCSupport.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"

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

mlir::LogicalResult VariantToEmitCFunc::emitGgmlQuantizeRowQ80(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The q8_0 activation quantizer is now FRONT-DOOR CONSTRUCTED (G3 line-B
    // quantize front door, family-head of the f32->QUANT spectrum): rather than
    // emit the hand-written monolith directly, CONSTRUCT the typed
    // weft_rvv.typed_quantize_row_loop_body region { quantize_row_encode_core;
    // typed_quantize_row_loop_yield } in place of the abstract
    // weft_rvv.quantize_row_q8_0 and LOWER it via emitTypedQuantizeRowLoopBody ->
    // the SHARED body emitQuantizeRowQ80BodyShared. The emission is DRIVEN by the
    // typed region op-identity + encode_model ([L-6]/[L-8] construction), byte-exact
    // to the retired dispatch-wired q8_0 monolith modulo only the source-op
    // provenance token.
    weftrvv::GgmlQuantizeRowQ80Op quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<weftrvv::GgmlQuantizeRowQ80Op>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    // block_q8_0 AoS facts (ggml-common.h:241-245 + QK8_0 = 32): the fp16 d at byte
    // 0, the 32 int8 qs at byte 2, stride 34. The per-format extra offsets baked into
    // the encode leaf (none for q8_0).
    return constructQuantizeRowRegionAndLower(
        rewriter, loc, scope, quantOp.getOperation(), quantOp.getInput(),
        quantOp.getOutput(), quantOp.getElementCount(), /*encodeModel=*/"q8_0",
        /*qk=*/32, /*stride=*/34, /*scaleOff=*/0, /*quantOff=*/2, avlArg,
        sizeType, valueMap);
  }

// The SHARED q8_0 quantize_row block-encode body: the AoS `nb = n/32` block loop,
// the per-block f32 load in ONE e32m8 strip, and the amax/scale/narrow q8_0 core
// (emitQuantizeQ80BlockBody). Extracted VERBATIM from the block-loop tail of the
// retired emitGgmlQuantizeRowQ80 monolith so the CONSTRUCTED typed lowering (via
// emitTypedQuantizeRowLoopBody) emits byte-identical C (modulo only the source-op
// provenance token threaded through opName/role). Byte-exact to ggml's EXACT RVV
// method (riscv/quants.c:32-71). Streaming (no cross-block accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitQuantizeRowQ80BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
    // block_q8_0 AoS facts: fp16 d @0, 32 int8 qs @2, stride 34.
    const int64_t qk = 32, blockStride = 34, scaleOffset = 0, quantOffset = 2;

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

mlir::LogicalResult VariantToEmitCFunc::emitGgmlQuantizeRowQ81(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The q8_1 SIBLING is now FRONT-DOOR CONSTRUCTED (G3 line-B quantize front
    // door): CONSTRUCT the typed weft_rvv.typed_quantize_row_loop_body region
    // (encode_model "q8_1") in place of the abstract weft_rvv.quantize_row_q8_1 and
    // LOWER it via emitTypedQuantizeRowLoopBody -> the SHARED body
    // emitQuantizeRowQ81BodyShared (the q8_0 amax/scale/narrow SIBLING + the extra
    // vwredsum integer block sum stored as the fp16 block_q8_1.s). Byte-exact to the
    // retired dispatch-wired q8_1 monolith modulo only the source-op provenance token.
    weftrvv::GgmlQuantizeRowQ81Op quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<weftrvv::GgmlQuantizeRowQ81Op>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    // block_q8_1 AoS facts (ggml-common.h:248-259 + QK8_1 = 32): the fp16 d at byte
    // 0, the fp16 s at byte 2, the 32 int8 qs at byte 4, stride 36. The sum byte
    // offset (2) is baked into the q8_1 encode leaf, so the brick carries only the
    // shared scale/quant offsets.
    return constructQuantizeRowRegionAndLower(
        rewriter, loc, scope, quantOp.getOperation(), quantOp.getInput(),
        quantOp.getOutput(), quantOp.getElementCount(), /*encodeModel=*/"q8_1",
        /*qk=*/32, /*stride=*/36, /*scaleOff=*/0, /*quantOff=*/4, avlArg,
        sizeType, valueMap);
  }

// The SHARED q8_1 quantize_row block-encode body: the q8_0 amax/scale/narrow SIBLING
// PLUS the extra vwredsum integer block sum stored as the fp16 block_q8_1.s.
// Extracted VERBATIM from the block-loop tail of the retired emitGgmlQuantizeRowQ81
// monolith so the CONSTRUCTED typed lowering (via emitTypedQuantizeRowLoopBody) emits
// byte-identical C (modulo only the source-op provenance token threaded through
// opName/role). Byte-exact to ggml's EXACT RVV method. Streaming (no cross-block
// accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitQuantizeRowQ81BodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
    // block_q8_1 AoS facts: fp16 d @0, fp16 s @2, 32 int8 qs @4, stride 36.
    const int64_t qk = 32, blockStride = 36, scaleOffset = 0, sumOffset = 2,
                  quantOffset = 4;

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

mlir::LogicalResult VariantToEmitCFunc::emitGgmlQuantizeRowQ8K(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The heaviest quantizer (ggml quantize_row_q8_K RVV path, riscv/quants.c):
    // a QK_K=256 super-block loop -- a min/max reduction, a symmetric iscale, the
    // vfcvt/vnclip RNE narrow, the FLOAT d store, the per-16 vwredsum bsums, and
    // the zero-block memset special case. Now FRONT-DOOR CONSTRUCTED (G3 line-B
    // quantize front door): CONSTRUCT the typed weft_rvv.typed_quantize_row_loop_body
    // region (encode_model "q8_K") in place of the abstract weft_rvv.quantize_row_q8_K
    // and LOWER it via emitTypedQuantizeRowLoopBody -> the SHARED body
    // emitQuantizeRowQ8KBodyShared. Byte-exact to the retired dispatch-wired q8_K
    // monolith modulo only the source-op provenance token.
    weftrvv::GgmlQuantizeRowQ8KOp quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<weftrvv::GgmlQuantizeRowQ8KOp>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    // block_q8_K AoS facts (ggml-common.h:360-366 + QK_K = 256): the FLOAT d at byte
    // 0, the 256 int8 qs at byte 4, the 16 int16 bsums at byte 260, stride 292. The
    // bsums byte offset (260) is baked into the q8_K encode leaf, so the brick
    // carries only the shared scale/quant offsets.
    return constructQuantizeRowRegionAndLower(
        rewriter, loc, scope, quantOp.getOperation(), quantOp.getInput(),
        quantOp.getOutput(), quantOp.getElementCount(), /*encodeModel=*/"q8_K",
        /*qk=*/256, /*stride=*/292, /*scaleOff=*/0, /*quantOff=*/4, avlArg,
        sizeType, valueMap);
  }

// The SHARED q8_K quantize_row super-block-encode body: the QK_K=256 min/max-
// symmetric quantizer -- the vsetvlmax_e32m8-folded min/max strip loop + scalar
// reduce, the fabsf-symmetric iscale, the STRUCTURED amax==0 zero path (float d=0,
// memset qs+bsums), else the FLOAT d store + the quantize strip (vfmul + vfcvt/vnclip
// RNE narrow) + the 256 int8 qs store + the 16 per-16 vslidedown-advanced vwredsum
// bsums. Extracted VERBATIM from the loop tail of the retired emitGgmlQuantizeRowQ8K
// monolith so the CONSTRUCTED typed lowering (via emitTypedQuantizeRowLoopBody) emits
// byte-identical C (modulo only the source-op provenance token threaded through
// opName/role). Byte-exact to ggml's EXACT RVV method. Streaming (no cross-block
// accumulator).
mlir::LogicalResult VariantToEmitCFunc::emitQuantizeRowQ8KBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
    // block_q8_K AoS facts: FLOAT d @0, 256 int8 qs @4, 16 int16 bsums @260,
    // stride 292.
    const int64_t qk = 256, blockStride = 292, scaleOffset = 0, quantOffset = 4,
                  bsumsOffset = 260;

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

  // Bounded encode_model surface gate (I7): the constructed streaming quantize
  // family is {q8_0 (family-head), q8_1 (the SIBLING + block sum), q8_K (the K-quant
  // activation quantizer)}. The verifier already gates encode_model; this fails the
  // emit closed if a not-yet-lowered encode leaf slips a valid-verify region here.
  llvm::StringRef encodeModel = loopBody.getEncodeModel();
  if (encodeModel != "q8_0" && encodeModel != "q8_1" && encodeModel != "q8_K")
    return rewriter.notifyMatchFailure(
        loopBody, "typed quantize_row loop body only lowers the constructed "
                  "streaming encode_models q8_0/q8_1/q8_K");

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
  // Dispatch on encode_model to the per-format leaf: each re-emits the whole nb
  // block loop + per-block encode via the SHARED body emitter, byte-exact to the
  // retired per-format monolith.
  if (encodeModel == "q8_1")
    return emitQuantizeRowQ81BodyShared(rewriter, loc, input, output, avlArg,
                                        sizeType, opName, role);
  if (encodeModel == "q8_K")
    return emitQuantizeRowQ8KBodyShared(rewriter, loc, input, output, avlArg,
                                        sizeType, opName, role);
  return emitQuantizeRowQ80BodyShared(rewriter, loc, input, output, avlArg,
                                      sizeType, opName, role);
}

// The quantize FRONT DOOR (the streaming activation-quantizer family {q8_0 family-head
// + the q8_1 sibling + the q8_K K-quant quantizer}): CONSTRUCT the typed
// weft_rvv.typed_quantize_row_loop_body region { quantize_row_encode_core;
// typed_quantize_row_loop_yield } in place of the abstract per-format
// weft_rvv.quantize_row_q8_{0,1,K}, then LOWER it via emitTypedQuantizeRowLoopBody. The
// construction is a genuine IR rewrite (the emission is DRIVEN by the typed region
// op-identity + encode_model, not the abstract op identity alone), so these formats are
// CONSTRUCTED ([L-6]/[L-8]), not dispatch-wired. The MIRROR of
// constructOrEmitGgmlDequantizeRow's construction half.
mlir::LogicalResult VariantToEmitCFunc::constructQuantizeRowRegionAndLower(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Operation *quantOp, mlir::Value input,
    mlir::Value output, mlir::Value n, llvm::StringRef encodeModel, int64_t qk,
    int64_t stride, int64_t scaleOff, int64_t quantOff, mlir::Value avlArg,
    mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  // CONSTRUCT the typed weft_rvv.typed_quantize_row_loop_body region in place of
  // the abstract quantOp via the SHARED byte-exact construction
  // (RVVQuantizeRowConstruction) so the pre-emitc RVVQuantizeRowStreamFrontDoor pass
  // and THIS in-emitc fallback build the IDENTICAL typed region. The per-format ABI
  // block facts (qk/stride/scale/quant) supplied by the emitGgmlQuantizeRowQ8{0,1,K}
  // entry points are the SAME facts lookupQuantizeRowStreamFacts returns, so the
  // region is byte-identical either way. Then LOWER it -- the emission is DRIVEN by
  // the typed region op-identity + encode_model ([L-6]/[L-8] construction).
  weftrvv::QuantizeRowStreamFacts facts{qk, stride, scaleOff, quantOff};
  if (mlir::failed(weftrvv::constructTypedQuantizeRowLoopBody(
          rewriter, quantOp, input, output, n, encodeModel, facts)))
    return mlir::failure();

  return emitTypedQuantizeRowLoopBody(rewriter, loc, scope, avlArg, sizeType,
                                      valueMap);
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

mlir::LogicalResult VariantToEmitCFunc::emitGgmlDequantizeRow(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // The DISPATCH-WIRED dequantize_row family body: the single
    // weft_rvv.dequantize_row op's bounded `format` routes to a hand-written AoS
    // block-decode ([L-6] wiring != construction). The decode is byte-exact to
    // ggml's reference dequantize_row_<format> (a scalar block loop); the per-strip
    // f32 stores need no reduction, so LMUL/strip-count are correctness-free.
    weftrvv::GgmlDequantizeRowOp deqOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto d = llvm::dyn_cast<weftrvv::GgmlDequantizeRowOp>(op))
        deqOp = d;
    }
    if (!deqOp)
      return rewriter.notifyMatchFailure(scope, "dequant body missing the op");

    mlir::Value input = valueMap.lookup(deqOp.getInput());
    mlir::Value output = valueMap.lookup(deqOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(deqOp, "dequant ABI operand unmapped");

    llvm::StringRef format = deqOp.getFormat();
    llvm::StringRef opName = deqOp.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = deqOp.getWEFTEmitCLowerableSourceRole();

    // The EXTENDED formats (K-quant super-blocks / FP4 codebooks / ternary /
    // iq4_nl) each route to their own hand-written scalar super-block decode
    // (byte-exact to ggml's reference dequantize_row_<format>, reusing that
    // format's already-constructed vec_dot block-decode facts). The extended
    // dispatch returns a plain match failure (no ops emitted) for a legacy
    // format, so the legacy nibble/int8 chain below still owns q4_0..q8_0.
    if (mlir::succeeded(emitGgmlDequantizeRowExtended(
            rewriter, loc, format, input, output, avlArg, sizeType, opName,
            role)))
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

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
    auto uintLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
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
    auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) {
      return emitLoadByteAsUint(rewriter, loc, constU8Type, uintType, ptr, i);
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
    int64_t stride, int64_t dOff, int64_t mOff, int64_t qhOff, int64_t qsOff,
    int64_t sub, bool hasMin, bool hasQh) const {
  // block_q4_0/q4_1/q5_0/q5_1 AoS facts: qk=32 lanes per block, 16 packed nibble bytes
  // (the byte-exact ggml ABI shape constants, NOT tunable knobs -- the SAME facts the
  // scalar emitDequantizeRowNibbleBodyShared hard-codes).
  const int64_t qk = 32, half = qk / 2;

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
  mlir::Value halfVl = sizeLit(half);

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

// The per-format OWNED REAL-VECTOR nibble leaves: each hard-codes its ggml block_qX AoS
// layout facts (the byte-exact ABI shape constants, NOT tunable knobs) and calls the
// SHARED vector body. Only the CONSTRUCTED path routes here; the monolith fallback keeps
// the scalar emitDequantizeRow<FMT>BodyShared. Siblings of emitDequantizeRowQ8_0VectorBody.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ4_0VectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleVectorBody(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/18, /*dOff=*/0, /*mOff=*/0, /*qhOff=*/0, /*qsOff=*/2,
      /*sub=*/8, /*hasMin=*/false, /*hasQh=*/false);
}
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ5_0VectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleVectorBody(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/22, /*dOff=*/0, /*mOff=*/0, /*qhOff=*/2, /*qsOff=*/6,
      /*sub=*/16, /*hasMin=*/false, /*hasQh=*/true);
}
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ4_1VectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleVectorBody(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/20, /*dOff=*/0, /*mOff=*/2, /*qhOff=*/0, /*qsOff=*/4,
      /*sub=*/0, /*hasMin=*/true, /*hasQh=*/false);
}
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowQ5_1VectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  return emitDequantizeRowNibbleVectorBody(
      rewriter, loc, input, output, avlArg, sizeType, opName, role,
      /*stride=*/24, /*dOff=*/0, /*mOff=*/2, /*qhOff=*/4, /*qsOff=*/8,
      /*sub=*/0, /*hasMin=*/true, /*hasQh=*/true);
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

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
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

// The OWNED REAL-VECTOR q8_0 dequantize_row block-decode body (PR-31, the FIRST
// non-grid cell of the dequant true-vector emitter): each block is 32 signed int8
// quants scaled by ONE fp16 `d`, so the whole per-block decode is a single 32-lane
// vector pipeline -- vle8 (the 32 int8 quants) + vsext_vf4 (int8->int32, the load
// sign-extends) + vfcvt_f_x_v (int32->f32) + vfmul_vf (the runtime `d` scale) + vse32
// (the contiguous 32-float store). NO gather (q8_0 is NON-GRID: no codebook, no sign
// plane, no nibble unpack), so unlike the iq3_xxs grid leaf there is no vluxei
// indexed-gather wall -- a clean streaming vector dequant. The vector content is the
// EMITTER's (OWNED __riscv_v intrinsics), NOT host-autovec codegen-lottery: the
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
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  // block_q8_0 AoS facts: fp16 d @0, 32 signed int8 quants @2, stride 34 (the
  // byte-exact ggml ABI shape constants, NOT tunable knobs -- the SAME facts the
  // scalar emitDequantizeRowQ8_0BodyShared hard-codes).
  const int64_t qk = 32, stride = 34, qsOff = 2;

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
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{xb}, opName, role,
                                   llvm::StringRef("fcvt.s.h"));

    // const int8_t *qs = (const int8_t *)(xb + 2);
    mlir::Value qsBaseRaw =
        rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff));
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

// ============================================================================
// The OWNED REAL-VECTOR K-quant super-block dequantize_row bodies (R线 §四.2, the
// K-quant fan-out over the q8_0/nibble non-grid precedent · de-lottery [L-8] ·
// ISSUE-001 reverse · closes the ISSUE-002 codegen-lottery exposure per format). Each
// mirrors the SAME integer quant decode the scalar emitGgmlDequantizeRowExtended runs
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
    bool isQ5) const {
  const int64_t qk = 256, stride = isQ5 ? 176 : 144, qsOff = isQ5 ? 48 : 16;
  const int64_t qhOff = 16; // q5_K only

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
  mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m8_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m8_t");
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
    mlir::Value dminAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(2)).getResult();
    mlir::Value dmin = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dminAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    // get_scale_min_k4(j) for compile-time j (0..7), scales at +4 (byte-exact).
    auto sc = [&](int k) { return loadU8Int(xb, 4 + k); };
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
                           mlir::ValueRange{qhU8, sizeLit(32)}, opName, role);
    }

    // Emit one 32-lane half-super-sub pipeline: nibblePlane (u8m2) -> [q5 5th-bit at
    // bit `bitShift` of qh] -> u32m8 -> i32m8 -> f32m8 -> fused d*v - ml -> vse32.
    auto emitHalf = [&](mlir::Value nibblePlane, int64_t bitShift, mlir::Value scaleD,
                        mlir::Value minML, mlir::Value yStore) {
      mlir::Value plane = nibblePlane;
      if (isQ5) {
        mlir::Value sh = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m2",
                                        mlir::ValueRange{qhv, u8Lit(bitShift), sizeLit(32)}, opName, role);
        mlir::Value bit = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m2",
                                         mlir::ValueRange{sh, u8Lit(1), sizeLit(32)}, opName, role);
        mlir::Value hb = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsll_vx_u8m2",
                                        mlir::ValueRange{bit, u8Lit(4), sizeLit(32)}, opName, role);
        plane = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vor_vv_u8m2",
                               mlir::ValueRange{plane, hb, sizeLit(32)}, opName, role);
      }
      mlir::Value n32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m8",
                                       mlir::ValueRange{plane, sizeLit(32)}, opName, role);
      mlir::Value ni = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m8_i32m8",
                                      mlir::ValueRange{n32}, opName, role);
      mlir::Value nF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m8", "f32"),
                                      mlir::ValueRange{ni, sizeLit(32)}, opName, role);
      mlir::Value acc = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmv_v_f", 32, "m8", "f32"),
                                       mlir::ValueRange{minML, sizeLit(32)}, opName, role);
      mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, "__riscv_vfmsac_vf_f32m8",
                                     mlir::ValueRange{acc, scaleD, nF, sizeLit(32)}, opName, role);
      emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m8", "f32"),
                         mlir::ValueRange{yStore, r, sizeLit(32)}, opName, role);
    };

    for (int64_t jj = 0; jj < 4; ++jj) {
      mlir::Value sc0, m0, sc1, m1v;
      scaleMin((int)(2 * jj), sc0, m0);
      scaleMin((int)(2 * jj + 1), sc1, m1v);
      mlir::Value d1 = fMul(d, i2f(sc0));
      mlir::Value ml1 = fMul(dmin, i2f(m0));
      mlir::Value d2 = fMul(d, i2f(sc1));
      mlir::Value ml2 = fMul(dmin, i2f(m1v));

      mlir::Value qsPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff + jj * 32)).getResult();
      mlir::Value qsU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qsPtr).getResult();
      mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m2", "u8"),
                                      mlir::ValueRange{qsU8, sizeLit(32)}, opName, role);

      mlir::Value nlo = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m2",
                                       mlir::ValueRange{qv, u8Lit(15), sizeLit(32)}, opName, role);
      mlir::Value yLo = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(jj * 64)).getResult();
      emitHalf(nlo, 2 * jj, d1, ml1, yLo);

      mlir::Value nhi = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m2",
                                       mlir::ValueRange{qv, u8Lit(4), sizeLit(32)}, opName, role);
      mlir::Value yHi = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(jj * 64 + 32)).getResult();
      emitHalf(nhi, 2 * jj + 1, d2, ml2, yHi);
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
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  const int64_t qk = 256, stride = 84;

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
    mlir::Value dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(80)).getResult();
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{dAddr}, opName, role, llvm::StringRef("fcvt.s.h"));
    mlir::Value dminAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(82)).getResult();
    mlir::Value dmin = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dminAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    for (int64_t nn = 0; nn < 2; ++nn) {
      int64_t qBlk = 16 + nn * 32;
      int64_t outBlk = nn * 128;
      for (int64_t j = 0; j < 4; ++j) {
        int64_t shift = 2 * j;
        int64_t is0 = nn * 8 + j * 2;
        int64_t outJ = outBlk + j * 32;
        for (int64_t half = 0; half < 2; ++half) {
          mlir::Value scv = loadU8Int(xb, is0 + half);
          mlir::Value dl = fMul(d, i2f(iAnd(scv, intLit(0xF))));
          mlir::Value ml = fMul(dmin, i2f(iShr(scv, intLit(4))));
          int64_t qHalf = qBlk + half * 16;
          int64_t outHalf = outJ + half * 16;

          mlir::Value qPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qHalf)).getResult();
          mlir::Value qU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qPtr).getResult();
          mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                          mlir::ValueRange{qU8, sizeLit(16)}, opName, role);
          mlir::Value shd = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                           mlir::ValueRange{qv, u8Lit(shift), sizeLit(16)}, opName, role);
          mlir::Value q2 = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                          mlir::ValueRange{shd, u8Lit(3), sizeLit(16)}, opName, role);
          mlir::Value q32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                           mlir::ValueRange{q2, sizeLit(16)}, opName, role);
          mlir::Value qi = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                          mlir::ValueRange{q32}, opName, role);
          mlir::Value qF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                          mlir::ValueRange{qi, sizeLit(16)}, opName, role);
          mlir::Value acc = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmv_v_f", 32, "m4", "f32"),
                                           mlir::ValueRange{ml, sizeLit(16)}, opName, role);
          mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, "__riscv_vfmsac_vf_f32m4",
                                         mlir::ValueRange{acc, dl, qF, sizeLit(16)}, opName, role);
          mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outHalf)).getResult();
          emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                             mlir::ValueRange{yStore, r, sizeLit(16)}, opName, role);
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
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  const int64_t qk = 256, stride = 110;

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
    mlir::Value dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(108)).getResult();
    mlir::Value dAll = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    // The derived 6-bit scale for compile-time is (0..15) from the 12 packed scale
    // bytes at +96 (ggml aux kmask1/kmask2 shuffle, byte-exact to the scalar decode).
    auto q3Scale = [&](int is) -> mlir::Value {
      int group = is / 4, b = is % 4;
      auto sb = [&](int k) { return loadU8Int(xb, 96 + k); };
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
      int64_t qBlk = 32 + nn * 32;
      int64_t outBlk = nn * 128;
      for (int64_t j = 0; j < 4; ++j) {
        int64_t shift = 2 * j;
        int64_t mbit = nn * 4 + j;
        int64_t outJ = outBlk + j * 32;
        for (int64_t half = 0; half < 2; ++half) {
          int is = (int)(nn * 8 + 2 * j + half);
          mlir::Value dl = fMul(dAll, i2f(iSub(q3Scale(is), intLit(32))));
          int64_t qHalf = qBlk + half * 16;
          int64_t hmHalf = half * 16;
          int64_t outHalf = outJ + half * 16;

          mlir::Value qPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qHalf)).getResult();
          mlir::Value qU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qPtr).getResult();
          mlir::Value qv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                          mlir::ValueRange{qU8, sizeLit(16)}, opName, role);
          mlir::Value hmPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(hmHalf)).getResult();
          mlir::Value hmU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, hmPtr).getResult();
          mlir::Value hmv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                           mlir::ValueRange{hmU8, sizeLit(16)}, opName, role);

          mlir::Value shd = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                           mlir::ValueRange{qv, u8Lit(shift), sizeLit(16)}, opName, role);
          mlir::Value qbits8 = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                              mlir::ValueRange{shd, u8Lit(3), sizeLit(16)}, opName, role);
          mlir::Value hmsh = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                            mlir::ValueRange{hmv, u8Lit(mbit), sizeLit(16)}, opName, role);
          mlir::Value bit8 = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                            mlir::ValueRange{hmsh, u8Lit(1), sizeLit(16)}, opName, role);

          mlir::Value qbits32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                               mlir::ValueRange{qbits8, sizeLit(16)}, opName, role);
          mlir::Value qbitsI = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                              mlir::ValueRange{qbits32}, opName, role);
          mlir::Value bit32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                             mlir::ValueRange{bit8, sizeLit(16)}, opName, role);
          mlir::Value bitI = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                            mlir::ValueRange{bit32}, opName, role);
          // term = (1 - bit) << 2 ; qdec = qbits - term.
          mlir::Value oneMinus = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vrsub_vx_i32m4",
                                                mlir::ValueRange{bitI, i32Lit(1), sizeLit(16)}, opName, role);
          mlir::Value term = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsll_vx_i32m4",
                                            mlir::ValueRange{oneMinus, i32Lit(2), sizeLit(16)}, opName, role);
          mlir::Value qdec = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsub_vv_i32m4",
                                            mlir::ValueRange{qbitsI, term, sizeLit(16)}, opName, role);
          mlir::Value qF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                          mlir::ValueRange{qdec, sizeLit(16)}, opName, role);
          mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, "m4", "f32"),
                                         mlir::ValueRange{qF, dl, sizeLit(16)}, opName, role);
          mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outHalf)).getResult();
          emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                             mlir::ValueRange{yStore, r, sizeLit(16)}, opName, role);
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
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  const int64_t qk = 256, stride = 210;

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
    mlir::Value dAddr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(208)).getResult();
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{dAddr}, opName, role, llvm::StringRef("fcvt.s.h"));

    for (int64_t nn = 0; nn < 2; ++nn) {
      int64_t qlBlk = nn * 64;
      int64_t qhBlk = 128 + nn * 32;
      int64_t scBlk = 192 + nn * 8;
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
                                           mlir::ValueRange{qlU8, sizeLit(16)}, opName, role);
          mlir::Value qhPtr = rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qhBlk + lbase)).getResult();
          mlir::Value qhU8 = rewriter.create<emitc::CastOp>(loc, u8PtrType, qhPtr).getResult();
          mlir::Value qhv = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                           mlir::ValueRange{qhU8, sizeLit(16)}, opName, role);

          mlir::Value nib;
          if (nibLo)
            nib = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                 mlir::ValueRange{qlv, u8Lit(0x0F), sizeLit(16)}, opName, role);
          else
            nib = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                 mlir::ValueRange{qlv, u8Lit(4), sizeLit(16)}, opName, role);
          mlir::Value qhsh = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                            mlir::ValueRange{qhv, u8Lit(qhShift), sizeLit(16)}, opName, role);
          mlir::Value qhbits = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                              mlir::ValueRange{qhsh, u8Lit(3), sizeLit(16)}, opName, role);
          mlir::Value qhhi = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsll_vx_u8m1",
                                            mlir::ValueRange{qhbits, u8Lit(4), sizeLit(16)}, opName, role);
          mlir::Value combined = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vor_vv_u8m1",
                                                mlir::ValueRange{nib, qhhi, sizeLit(16)}, opName, role);
          mlir::Value c32 = emitOpaqueCall(rewriter, loc, u32VecType, "__riscv_vzext_vf4_u32m4",
                                           mlir::ValueRange{combined, sizeLit(16)}, opName, role);
          mlir::Value ci = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vreinterpret_v_u32m4_i32m4",
                                          mlir::ValueRange{c32}, opName, role);
          mlir::Value qi = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsub_vx_i32m4",
                                          mlir::ValueRange{ci, i32Lit(32), sizeLit(16)}, opName, role);
          mlir::Value qF = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                          mlir::ValueRange{qi, sizeLit(16)}, opName, role);
          mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, "m4", "f32"),
                                         mlir::ValueRange{qF, dsc, sizeLit(16)}, opName, role);
          mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outOff)).getResult();
          emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
                             mlir::ValueRange{yStore, r, sizeLit(16)}, opName, role);
        }
      }
    }
  }
  return mlir::success();
}

// The SCALAR K-quant super-block dequantize_row forwarder (q2_K/q3_K/q4_K/q5_K/q6_K):
// a thin FORWARDER to the SAME hand-written super-block decode the dispatch-wired
// monolith fallback runs (emitGgmlDequantizeRowExtended, keyed by the `format` string
// alone -- no deqOp). The CONSTRUCTED typed dispatch now routes K-quant to the OWNED
// REAL-VECTOR bodies (emitDequantizeRowQ45KVectorBody / ...Q2K / ...Q3K / ...Q6K, R线
// §四.2), so this scalar forwarder is OFF the hot dispatch; it is retained as the
// documented scalar-path entry point, and the scalar decode it wraps
// (emitGgmlDequantizeRowExtended) is still the monolith fallback that owned-vector emit
// is byte-exact against. Streaming sibling of the flat leaves
// (emitDequantizeRowQ4_0BodyShared etc.); no reduction / no accumulator.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowKQuantBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef format) const {
  return emitGgmlDequantizeRowExtended(rewriter, loc, format, input, output,
                                       avlArg, sizeType, opName, role);
}

// The per-format CONSTRUCTED dequantize_row decode leaf for the QK_K=256 IQ
// grid-table super-block family (iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s): a thin
// FORWARDER to the SAME hand-written grid-decode the dispatch-wired monolith fallback
// runs (emitGgmlDequantizeRowExtended, keyed by the `format` string alone -- no
// deqOp). Each IQ grid leaf emits its canonical grid + sign-plane table decls as
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
    llvm::StringRef format) const {
  // iq3_xxs is the FIRST cell of the dequant true-vector emitter (PR-31): it lowers
  // to the OWNED real-vector body (vluxei16 grid gather + sign fold + vfcvt + vfmul
  // + vse32), NOT the shared scalar dispatch-wired decode. The other IQ grid formats
  // (iq2_xxs/iq2_xs/iq2_s/iq3_s) stay on the scalar forwarder until they fan out.
  if (format == "iq3_xxs")
    return emitDequantizeRowIQ3XXSVectorBody(rewriter, loc, input, output, avlArg,
                                             sizeType, opName, role);
  return emitGgmlDequantizeRowExtended(rewriter, loc, format, input, output,
                                       avlArg, sizeType, opName, role);
}

// The per-format CONSTRUCTED dequantize_row decode leaf for the remaining codebook /
// ternary-grid extended formats: the ternary iq1s_grid leaves iq1_s (fp16 d + qh scale
// + delta) / iq1_m (reconstructed packed iq1m_scale + per-group delta, NO fp16 d), the
// 16-entry non-linear codebook leaves iq4_nl (flat fp16 scale) / iq4_xs (super-block
// signed-6 scale), and the FP4 e2m1 codebook leaves mxfp4 (E8M0 shared exponent) /
// nvfp4 (four UE4M3 sub-block scales). A thin FORWARDER to the SAME hand-written
// extended decode the dispatch-wired monolith fallback runs (emitGgmlDequantizeRowExtended,
// keyed by the `format` string alone -- no deqOp). Each leaf emits its codebook /
// ternary grid table as function-local statics (the SAME emitIQ1SCanonicalGridTableDecl /
// emitIQ1MCanonicalGridTableDecl / emitCodebookDecl anchors the block-dot vec_dot
// lowerings render) then a scalar AoS block loop -- there is NO op-attribute dependency
// (the grid / codebook / E8M0-UE4M3 scale planes are DERIVED at emit, not carried as
// op-attrs), so the leaf is self-contained and CLEANLY constructible. Because BOTH the
// monolith fallback AND this constructed lowering emit the decode from the SAME code, the
// two are byte-exact by construction (modulo only the source-op provenance token threaded
// through opName/role) -- there is NO duplicated decode leaf to drift. Streaming sibling
// of emitDequantizeRowIQGridBodyShared; no reduction / no accumulator.
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowCodebookGridBodyShared(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef format) const {
  // The four 16-entry codebook leaves (mxfp4/nvfp4 FP4 e2m1, iq4_nl/iq4_xs non-linear)
  // lower to the OWNED vrgather codebook vector body (B线批2 tiny-codebook de-lottery,
  // the FP4/non-linear fan-out over the q8_0/nibble non-grid precedent · [L-8] ·
  // closes the ISSUE-002 codegen-lottery for these formats).
  if (format == "mxfp4" || format == "nvfp4" || format == "iq4_nl" ||
      format == "iq4_xs")
    return emitDequantizeRowCodebookVectorBody(rewriter, loc, input, output,
                                               avlArg, sizeType, opName, role,
                                               format);
  // The tq1_0/tq2_0 base-3 / 2-bit ternary super-blocks lower to the OWNED ternary
  // ARITHMETIC vector body (B线批3 ternary de-lottery · [L-8] · closes the ISSUE-002
  // codegen-lottery for these formats). NO codebook table and NO gather -- the ternary
  // {-1,0,1} value is decoded by pure arithmetic (2-bit shift/mask; base-3 * pow3).
  // The ternary iq1_s/iq1_m stay on the scalar forwarder (they are a 2048-grid gather,
  // NOT a register-resident decode -- the gather-wall exclusion).
  if (format == "tq1_0" || format == "tq2_0")
    return emitDequantizeRowTernaryVectorBody(rewriter, loc, input, output,
                                              avlArg, sizeType, opName, role,
                                              format);
  return emitGgmlDequantizeRowExtended(rewriter, loc, format, input, output,
                                       avlArg, sizeType, opName, role);
}

// ============================================================================
// The OWNED REAL-VECTOR tiny-codebook (16-entry) dequantize_row body (B线批2, the FP4 /
// non-linear codebook fan-out over the q8_0/nibble non-grid precedent · de-lottery [L-8]
// · ISSUE-001 reverse · closes the ISSUE-002 codegen-lottery exposure per format). The
// 16-entry int8 codebook is broadcast into ONE i8m1 vreg ONCE; every group's two nibble
// index lanes are gathered through it (vrgather_vv_i8m1 -- a REGISTER-RESIDENT gather,
// NOT a vluxei memory gather -> NO HW-gather wall), sign-extended, int->float, scaled by
// the per-group float scale in ONE vfmul (== ggml's single `d*kv` mul -> no
// fp-contraction ambiguity), and stored. The integer nibble/codebook/scale decode
// mirrors the scalar emitGgmlDequantizeRowExtended byte-for-byte. Byte-exact to
// dequantize_row_{mxfp4,nvfp4,iq4_nl,iq4_xs} by construction.
// ============================================================================
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowCodebookVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef format) const {
  const bool isMx = format == "mxfp4";
  const bool isNl = format == "iq4_nl";
  const bool isNv = format == "nvfp4";
  const bool isXs = format == "iq4_xs";
  // Per-format super-block geometry (byte-exact ggml block_qX AoS facts, NOT knobs).
  const int64_t qk = isMx ? 32 : isNl ? 32 : isNv ? 64 : 256;
  const int64_t stride = isMx ? 17 : isNl ? 18 : isNv ? 36 : 136;

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
  mlir::Type u8VecType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
  mlir::Type i8VecType = emitc::OpaqueType::get(ctx, "vint8m1_t");
  mlir::Type i32VecType = emitc::OpaqueType::get(ctx, "vint32m4_t");
  mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m4_t");
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
  auto addSz = [&](mlir::Value a, mlir::Value b) {
    return rewriter.create<emitc::AddOp>(loc, sizeType, a, b).getResult();
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
  // byte-identical to the scalar emitGgmlDequantizeRowExtended nvfp4 seam.
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
  llvm::StringRef codebookName = (isMx || isNv) ? "weft_dequant_mxfp4_kvalues"
                                                : "weft_dequant_iq4nl_kvalues";
  {
    llvm::ArrayRef<int> entries = (isMx || isNv) ? llvm::ArrayRef<int>(kvaluesMxfp4)
                                                 : llvm::ArrayRef<int>(kvaluesIq4nl);
    std::string decl = "static const int8_t " + codebookName.str() + "[16] = {";
    for (size_t i = 0; i < entries.size(); ++i) {
      if (i) decl += ", ";
      decl += std::to_string(entries[i]);
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);
  }

  // vint8m1_t values = __riscv_vle8_v_i8m1(<codebook>, 16);  (broadcast the codebook
  // into ONE vreg, reused by every vrgather -- register-resident, NO memory gather).
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "codebook_table_load"));
  std::string tableLoadCallee = riscvIntrinsicName("vle", 8, "m1", "i8");
  mlir::Value values = emitOpaqueCallBuilt(
      rewriter, loc, i8VecType, tableLoadCallee, opName, role,
      [&](mlir::OpBuilder &, mlir::Location) -> llvm::SmallVector<mlir::Value> {
        mlir::Value tbl = rewriter.create<emitc::LiteralOp>(loc, i8PtrType, codebookName);
        return {tbl, sizeLit(16)};
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
      mlir::Value w = emitOpaqueCall(rewriter, loc, u8VecType, riscvIntrinsicName("vle", 8, "m1", "u8"),
                                     mlir::ValueRange{qsU8, sizeLit(nLanes)}, opName, role);
      mlir::Value idxLo = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vand_vx_u8m1",
                                         mlir::ValueRange{w, intLit(0x0F), sizeLit(nLanes)}, opName, role);
      mlir::Value idxHi = emitOpaqueCall(rewriter, loc, u8VecType, "__riscv_vsrl_vx_u8m1",
                                         mlir::ValueRange{w, intLit(4), sizeLit(nLanes)}, opName, role);
      auto lane = [&](mlir::Value idx, int64_t outOff) {
        mlir::Value g = emitOpaqueCall(rewriter, loc, i8VecType, "__riscv_vrgather_vv_i8m1",
                                       mlir::ValueRange{values, idx, sizeLit(nLanes)}, opName, role);
        mlir::Value w32 = emitOpaqueCall(rewriter, loc, i32VecType, "__riscv_vsext_vf4_i32m4",
                                         mlir::ValueRange{g, sizeLit(nLanes)}, opName, role);
        mlir::Value f = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfcvt_f_x_v", 32, "m4", "f32"),
                                       mlir::ValueRange{w32, sizeLit(nLanes)}, opName, role);
        mlir::Value r = emitOpaqueCall(rewriter, loc, f32VecType, riscvIntrinsicName("vfmul_vf", 32, "m4", "f32"),
                                       mlir::ValueRange{f, scale, sizeLit(nLanes)}, opName, role);
        mlir::Value yStore = rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outOff)).getResult();
        emitOpaqueCallVoid(rewriter, loc, riscvIntrinsicName("vse", 32, "m4", "f32"),
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
      emitHalfGroup(/*qsByteOff=*/isMx ? 1 : 2, /*nLanes=*/16, d, /*outLo=*/0, /*outHi=*/16);
    } else if (isNv) {
      // Four 16-element UE4M3-scaled sub-blocks; qs @+4, 8 bytes/sub. Per sub s:
      // y[s*16 + j] = d[s]*kv[qs[s*8+j]&0xF]; y[s*16+8 + j] = d[s]*kv[qs[s*8+j]>>4].
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "nvfp4_sub_decode"));
      for (int64_t s = 0; s < 4; ++s) {
        mlir::Value d = ue4m3ScaleAt(xb, s);
        emitHalfGroup(/*qsByteOff=*/4 + s * 8, /*nLanes=*/8, d, /*outLo=*/s * 16, /*outHi=*/s * 16 + 8);
      }
    } else { // iq4_xs
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
        emitHalfGroup(/*qsByteOff=*/8 + ib32 * 16, /*nLanes=*/16, dl,
                      /*outLo=*/ib32 * 32, /*outHi=*/ib32 * 32 + 16);
      }
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
//                  emitGgmlDequantizeRowExtended byte-for-byte; the ONE vfmul by d ==
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
    llvm::StringRef format) const {
  const bool isTq1 = format == "tq1_0";
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

  // Bounded decode_model surface gate (I7): the constructed streaming family is
  // {q8_0 (family-head), q4_0/q4_1/q5_0/q5_1 (flat nibble leaves), the
  // q2_K/q3_K/q4_K/q5_K/q6_K QK_K=256 super-block leaves, the
  // iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s QK_K=256 IQ grid-table super-block leaves,
  // and the remaining codebook / ternary-grid extended leaves iq1_s/iq1_m (ternary
  // iq1s_grid + delta), iq4_nl/iq4_xs (16-entry non-linear codebook), and mxfp4/nvfp4
  // (FP4 e2m1 codebook, E8M0 / UE4M3 scales)}. The verifier already gates
  // decode_model; this fails the emit closed if a not-yet-lowered decode leaf slips a
  // valid-verify region here.
  llvm::StringRef decodeModel = loopBody.getDecodeModel();
  if (decodeModel != "q8_0" && decodeModel != "q4_0" && decodeModel != "q4_1" &&
      decodeModel != "q5_0" && decodeModel != "q5_1" && decodeModel != "q1_0" &&
      decodeModel != "q2_K" &&
      decodeModel != "q3_K" && decodeModel != "q4_K" && decodeModel != "q5_K" &&
      decodeModel != "q6_K" && decodeModel != "iq2_xxs" &&
      decodeModel != "iq2_xs" && decodeModel != "iq2_s" &&
      decodeModel != "iq3_xxs" && decodeModel != "iq3_s" &&
      decodeModel != "iq1_s" && decodeModel != "iq1_m" &&
      decodeModel != "iq4_nl" && decodeModel != "iq4_xs" &&
      decodeModel != "mxfp4" && decodeModel != "nvfp4" &&
      decodeModel != "tq1_0" && decodeModel != "tq2_0")
    return rewriter.notifyMatchFailure(
        loopBody, "typed dequantize_row loop body only lowers the constructed "
                  "streaming decode_models q8_0/q4_0/q4_1/q5_0/q5_1 + the flat "
                  "binary-sign leaf q1_0 + the K-quant "
                  "super-blocks q2_K/q3_K/q4_K/q5_K/q6_K + the IQ grid-table "
                  "super-blocks iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s + the codebook / "
                  "ternary-grid leaves iq1_s/iq1_m/iq4_nl/iq4_xs/mxfp4/nvfp4 + the "
                  "ternary super-blocks tq1_0/tq2_0");

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
  // Dispatch on decode_model to the per-format leaf: each re-emits the whole nb
  // block loop + per-block decode via the SHARED body emitter, byte-exact to the
  // dispatch-wired monolith.
  // The flat nibble leaves (q4_0/q5_0 the single-mul SAFE set, q4_1/q5_1 the min-add
  // FMA set) are the THIRD..SIXTH cells of the dequant true-vector emitter (R线 §四.1,
  // fan-out over the q8_0 non-grid precedent): the CONSTRUCTED path lowers to the OWNED
  // real-vector body (vle8 + vand/vsrl nibble split + vzext + [q5 5th-bit spread] +
  // vfcvt + vfmul_vf / [q4_1/q5_1 fused vfmacc_vf] + vse32, NO gather), NOT the scalar
  // per-element loop the dispatch-wired monolith fallback still runs. Byte-exact to the
  // ggml reference by construction (q4_0/q5_0 single-mul -> no fp-contraction ambiguity;
  // q4_1/q5_1 single fused mul-add -> matches the contracted opponent autovec vfmadd).
  // The monolith fallback keeps the scalar emitDequantizeRow<FMT>BodyShared.
  if (decodeModel == "q4_0")
    return emitDequantizeRowQ4_0VectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  if (decodeModel == "q4_1")
    return emitDequantizeRowQ4_1VectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  if (decodeModel == "q5_0")
    return emitDequantizeRowQ5_0VectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  if (decodeModel == "q5_1")
    return emitDequantizeRowQ5_1VectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  // The K-quant QK_K=256 super-block leaves (q2_K/q3_K/q4_K/q5_K/q6_K) are the R线
  // §四.2 K-quant fan-out cells of the dequant true-vector emitter: the CONSTRUCTED
  // path lowers to the OWNED real-vector body (per-super-sub vle8 + vand/vsrl bit
  // unpack + [q5_K/q6_K high-bit merge] + vzext_vf4 + vfcvt + fused vfmsac_vf (min
  // formats) / vfmul_vf (single-mul formats) + vse32, NO gather), NOT the scalar
  // per-element loop the dispatch-wired monolith fallback still runs. Byte-exact to
  // the ggml reference by construction (the integer quant decode mirrors the scalar
  // reference; q4_K/q5_K/q2_K fold `d1*v - m1` in ONE fused mul-sub matching the
  // contracted opponent autovec, q3_K/q6_K fold with a SINGLE mul -> no
  // fp-contraction ambiguity). The dispatch-wired monolith fallback keeps the scalar
  // K-quant decode (emitGgmlDequantizeRowExtended, the thin emitDequantizeRowKQuantBodyShared
  // forwarder's target) -- the q8_0/nibble precedent.
  if (decodeModel == "q4_K" || decodeModel == "q5_K")
    return emitDequantizeRowQ45KVectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role,
                                           /*isQ5=*/decodeModel == "q5_K");
  if (decodeModel == "q2_K")
    return emitDequantizeRowQ2KVectorBody(rewriter, loc, weightBase, output,
                                          avlArg, sizeType, opName, role);
  if (decodeModel == "q3_K")
    return emitDequantizeRowQ3KVectorBody(rewriter, loc, weightBase, output,
                                          avlArg, sizeType, opName, role);
  if (decodeModel == "q6_K")
    return emitDequantizeRowQ6KVectorBody(rewriter, loc, weightBase, output,
                                          avlArg, sizeType, opName, role);
  // The QK_K=256 IQ grid-table super-block leaves (iq2_xxs/iq2_xs/iq2_s/iq3_xxs/
  // iq3_s) forward to the SAME hand-written grid-decode the dispatch-wired monolith
  // runs, so the constructed emit is byte-exact to the monolith by construction.
  if (decodeModel == "iq2_xxs" || decodeModel == "iq2_xs" ||
      decodeModel == "iq2_s" || decodeModel == "iq3_xxs" ||
      decodeModel == "iq3_s")
    return emitDequantizeRowIQGridBodyShared(rewriter, loc, weightBase, output,
                                             avlArg, sizeType, opName, role,
                                             decodeModel);
  // The remaining codebook / ternary-grid extended leaves (iq1_s/iq1_m ternary
  // iq1s_grid, iq4_nl/iq4_xs non-linear codebook, mxfp4/nvfp4 FP4 codebook, and the
  // tq1_0/tq2_0 base-3 / 2-bit ternary super-blocks) forward to the SAME hand-written
  // extended decode the dispatch-wired monolith runs, so the constructed emit is
  // byte-exact to the monolith by construction.
  if (decodeModel == "iq1_s" || decodeModel == "iq1_m" ||
      decodeModel == "iq4_nl" || decodeModel == "iq4_xs" ||
      decodeModel == "mxfp4" || decodeModel == "nvfp4" ||
      decodeModel == "tq1_0" || decodeModel == "tq2_0")
    return emitDequantizeRowCodebookGridBodyShared(rewriter, loc, weightBase,
                                                   output, avlArg, sizeType,
                                                   opName, role, decodeModel);
  // The flat 1-bit binary-sign leaf (q1_0) forwards to the SAME hand-written
  // binary-sign decode the dispatch-wired monolith fallback runs (via the shared
  // emitGgmlDequantizeRowExtended, keyed by the format string alone), so the
  // constructed emit is byte-exact to the monolith by construction.
  if (decodeModel == "q1_0")
    return emitGgmlDequantizeRowExtended(rewriter, loc, decodeModel, weightBase,
                                         output, avlArg, sizeType, opName, role);
  // q8_0 is the SECOND cell of the dequant true-vector emitter (PR-31, the first
  // NON-GRID cell): the CONSTRUCTED path lowers to the OWNED real-vector body
  // (vle8 + vsext_vf4 + vfcvt + vfmul_vf + vse32, NO gather), NOT the scalar
  // per-element loop the dispatch-wired monolith fallback still runs. Byte-exact to
  // ggml's dequantize_row_q8_0 by construction (q8_0 has no add/min => no
  // fp-contraction ambiguity). The iq3_xxs precedent: the vector body lives only on
  // the constructed leaf; the monolith fallback keeps the scalar shared body.
  if (decodeModel == "q8_0")
    return emitDequantizeRowQ8_0VectorBody(rewriter, loc, weightBase, output,
                                           avlArg, sizeType, opName, role);
  return emitDequantizeRowQ8_0BodyShared(rewriter, loc, weightBase, output,
                                         avlArg, sizeType, opName, role);
}

// The dequant FRONT DOOR (the flat streaming family {q8_0 family-head + the
// q4_0/q4_1/q5_0/q5_1 nibble leaves} + the QK_K=256 K-quant super-block family
// {q2_K/q3_K/q4_K/q5_K/q6_K}): CONSTRUCT the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core;
// typed_dequantize_row_loop_yield } in place of the abstract weft_rvv.dequantize_row,
// then LOWER it via emitTypedDequantizeRowLoopBody. The construction is a genuine IR
// rewrite (the emission is DRIVEN by the typed region op-identity + decode_model, not
// the abstract format string), so these formats are CONSTRUCTED ([L-6]/[L-8]), not
// dispatch-wired. Plus the QK_K=256 IQ grid-table super-block family
// {iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s} (the fp16 d seam + the grid-of-N codebook
// gather + the per-format sign plane -- signs64/signs256/ksigns/per-lane sign bytes)
// and the remaining codebook / ternary-grid extended leaves {iq1_s/iq1_m (ternary
// iq1s_grid + delta), iq4_nl/iq4_xs (16-entry non-linear codebook), mxfp4/nvfp4 (FP4
// e2m1 codebook, E8M0 / UE4M3 scales), tq1_0/tq2_0 (the base-3 / 2-bit ternary
// super-blocks)}. Every modeled dequantize_row format is now CONSTRUCTED; only an
// unrecognized format falls through to the dispatch-wired monolith.
mlir::LogicalResult VariantToEmitCFunc::constructOrEmitGgmlDequantizeRow(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  weftrvv::GgmlDequantizeRowOp deqOp;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto d = llvm::dyn_cast<weftrvv::GgmlDequantizeRowOp>(op))
      deqOp = d;
  }
  if (!deqOp)
    return rewriter.notifyMatchFailure(scope, "dequant body missing the op");

  // The streaming CONSTRUCTED family per-format AoS block-layout facts now live in
  // the SHARED byte-exact construction (RVVDequantizeRowConstruction) so the
  // pre-emitc RVVDequantizeRowStreamFrontDoor pass and THIS in-emitc fallback build
  // the IDENTICAL typed region. Every modeled format now has facts; only an
  // unrecognized format (lookup == nullopt) stays DISPATCH-WIRED (the monolith).
  std::optional<weftrvv::DequantizeRowStreamFacts> facts =
      weftrvv::lookupDequantizeRowStreamFacts(deqOp.getFormat());
  if (!facts)
    return emitGgmlDequantizeRow(rewriter, loc, scope, avlArg, sizeType,
                                 valueMap);

  // CONSTRUCT the typed weft_rvv.typed_dequantize_row_loop_body region in place of
  // the abstract deqOp (the SAME construction the pre-emitc front door runs), then
  // LOWER it -- the emission is DRIVEN by the typed region op-identity + decode_model
  // ([L-6]/[L-8] construction), byte-exact to the retired monolith modulo only the
  // source-op provenance token.
  if (mlir::failed(weftrvv::constructTypedDequantizeRowLoopBody(rewriter, deqOp,
                                                               *facts)))
    return mlir::failure();

  return emitTypedDequantizeRowLoopBody(rewriter, loc, scope, avlArg, sizeType,
                                        valueMap);
}

// The EXTENDED dequantize_row decode: K-quant super-blocks, FP4 codebooks,
// ternary, and iq4_nl. Each body is a scalar AoS super-block loop reproducing
// ggml's reference dequantize_row_<format> byte-exactly (quants.c), reusing the
// SAME per-format block-decode facts already constructed for that format's
// block-dot vec_dot (the fp16 seam, the E8M0/UE4M3 scale, the get_scale_min_k4
// 6-bit unpack, the q3_K aux 6-bit scale shuffle, the base-3 tq1_0 unpack, the
// 16-entry codebook gather). Keyed by the `format` string alone (NO deqOp) so the
// K-quant super-blocks (q2_K/q3_K/q4_K/q5_K/q6_K) -- now FRONT-DOOR CONSTRUCTED --
// reach this SAME super-block decode from the constructed typed lowering (via
// emitDequantizeRowKQuantBodyShared), byte-identical to the dispatch-wired monolith
// fallback by construction; the NON-K-quant extended formats stay [L-6] wiring.
mlir::LogicalResult VariantToEmitCFunc::emitGgmlDequantizeRowExtended(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::StringRef format, mlir::Value input, mlir::Value output,
    mlir::Value avlArg, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role) const {
  enum Fmt {
    Q2K, Q3K, Q4K, Q5K, Q6K, MXFP4, NVFP4, TQ1, TQ2, IQ4NL,
    // The flat 1-bit binary-sign leaf (block_q1_0).
    Q10,
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
                .Case("q1_0", Q10)
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

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
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
  auto uLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
  auto uAnd = [&](mlir::Value a, mlir::Value b) { return emitBitAnd(rewriter, loc, uintType, a, b); };
  auto uOr = [&](mlir::Value a, mlir::Value b) { return emitBitOr(rewriter, loc, uintType, a, b); };
  auto uShl = [&](mlir::Value a, mlir::Value b) { return emitBitShl(rewriter, loc, uintType, a, b); };
  auto uShr = [&](mlir::Value a, mlir::Value b) { return emitBitShr(rewriter, loc, uintType, a, b); };
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
    codebookName = "weft_dequant_mxfp4_kvalues";
    emitCodebookDecl(codebookName, kvaluesMxfp4);
  } else if (fmt == NVFP4) {
    codebookName = "weft_dequant_nvfp4_kvalues";
    emitCodebookDecl(codebookName, kvaluesMxfp4);
  } else if (fmt == IQ4NL) {
    codebookName = "weft_dequant_iq4nl_kvalues";
    emitCodebookDecl(codebookName, kvaluesIq4nl);
  }
  if (fmt == TQ1) {
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t weft_dequant_tq1_0_pow3[6] = "
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
    codebookName = "weft_dequant_iq4nl_kvalues";
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
  case Q10: qk = 128; stride = 18; break;
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
      llvm::StringRef pow3 = "weft_dequant_tq1_0_pow3";
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
    } else if (fmt == Q10) {
      // Flat 1-bit binary-sign leaf (block_q1_0: fp16 d @0, qs[16] @2, qk=128).
      // ggml dequantize_row_q1_0: d = fp16(x.d); neg_d = -d; for j in 0..128:
      //   bit = (qs[j/8] >> (j%8)) & 1; y[j] = bit ? d : neg_d.
      mlir::Value d = fp16ReadAt(xb, 0);
      mlir::Value negD =
          rewriter.create<emitc::UnaryMinusOp>(loc, floatType, d).getResult();
      mlir::Value zeroI = intLit(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "q1_0_binary_sign_decode"));
      // qk/8 = 16 packed bytes x 8 bits = 128 lanes, output in ASCENDING j order.
      for (int64_t jb = 0; jb < qk / 8; ++jb) {
        mlir::Value qi =
            loadIntAt(xb, inputPtrType, sizeLit(2 + jb), constU8Type);
        for (int64_t bo = 0; bo < 8; ++bo) {
          mlir::Value bit = iAnd(iShr(qi, intLit(bo)), intLit(1));
          mlir::Value isSet = rewriter
                                  .create<emitc::CmpOp>(loc, boolType,
                                                        emitc::CmpPredicate::ne,
                                                        bit, zeroI)
                                  .getResult();
          mlir::Value val =
              rewriter
                  .create<emitc::ConditionalOp>(loc, floatType, isSet, d, negD)
                  .getResult();
          storeF32At(yb, sizeLit(jb * 8 + bo), val);
        }
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
      mlir::Value grid = castI8Base("weft_iq1s_grid", u64PtrType);
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
      mlir::Value grid = castI8Base("weft_iq1m_grid", u64PtrType);
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
      mlir::Value grid = castI8Base("weft_iq2xxs_grid", i64PtrType);
      mlir::Value signs = plainBase("weft_iq2xxs_signs64", i8PtrType);
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
      mlir::Value grid = castI8Base("weft_iq2xs_grid", i64PtrType);
      mlir::Value signs = plainBase("weft_iq2xs_signs64", i8PtrType);
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
      mlir::Value grid = castI8Base("weft_iq2s_grid", i64PtrType);
      mlir::Value signs = plainBase("weft_iq2s_signs256", i8PtrType);
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
      mlir::Value grid = castI8Base("weft_iq3xxs_grid", u32PtrType);
      mlir::Value ksigns = plainBase("weft_iq3xxs_ksigns", u8PtrType);
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
      mlir::Value grid = castI8Base("weft_iq3s_grid", u32PtrType);
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
} // namespace weft
