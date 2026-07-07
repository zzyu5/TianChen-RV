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

      // const float *xp = x + i;  float *yp = y + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, normIdx);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();
      mlir::Value yPtr =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, outputBuf, normIdx);
      mlir::Value yStorePtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();

      // vfloat32m<L>_t vx = __riscv_vle32_v_f32m<L>(xp, vl);
      std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
      mlir::Value vx = emitOpaqueCall(rewriter, loc, f32VecType, loadCallee,
                                      mlir::ValueRange{xLoadPtr, bodyVL}, opName,
                                      role);

      // vfloat32m<L>_t vy = __riscv_vfmul_vf_f32m<L>(vx, scale, vl);
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      mlir::Value vy = emitOpaqueCall(rewriter, loc, f32VecType, mulCallee,
                                      mlir::ValueRange{vx, scale, bodyVL}, opName,
                                      role);

      // __riscv_vse32_v_f32m<L>(yp, vy, vl);
      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");
      emitOpaqueCallVoid(rewriter, loc, storeCallee,
                         mlir::ValueRange{yStorePtr, vy, bodyVL}, opName, role);
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
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type half16Type = emitc::OpaqueType::get(ctx, "_Float16");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type half16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "_Float16"));
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "int8_t"));
    // The block's 32 lanes ride in ONE e32m8 strip (ggml's `size_t vl = QK8_0`);
    // the wide f32 / m1 reduce / i16m4 / i8m2 types are ggml's exact path types.
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
    mlir::Type f32m1Type = emitc::OpaqueType::get(ctx, "vfloat32m1_t");
    mlir::Type i16m4Type = emitc::OpaqueType::get(ctx, "vint16m4_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");

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

      // The amax reduction (ggml riscv/quants.c:47-50): vfabs -> vfredmax seeded
      // with a 0.0f f32m1 -> extract lane 0. The intrinsic callees are HARD-CODED
      // (ggml's exact spellings) -- never synthesized.
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

      // float d = amax / 127.0f;  (ggml's `amax / ((1 << 7) - 1)`; the divisor
      // is the f32 literal 127.0f so the divide is a single f32 round).
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
      mlir::Value c127 =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "127.0f");
      mlir::Value d = rewriter.create<emitc::DivOp>(loc, floatType, amax, c127);

      // float id = 0.0f; if (d != 0.0f) { id = 1.0f / d; }  -- the load-bearing
      // `id = d ? 1.0f/d : 0.0f` conditional, STRUCTURED (emitc.cmp + emitc.if,
      // NOT a raw string). The all-zero block (amax=0 => d=0) takes the else and
      // keeps id=0, so every q=0 (a bare 1/d would give inf, then 0*inf=NaN).
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

      // *(_Float16 *)(yb + 0) = (_Float16)d;  -- the fp16 d store. The board is
      // __riscv_zfhmin, so GGML_CPU_FP32_TO_FP16(d) is the native (_Float16)d
      // cast (fcvt.h.s, rne). STRUCTURED: cast the byte cursor to _Float16 *,
      // subscript [0] (an lvalue), cast d to _Float16, assign.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp16_d_store"));
      mlir::Value dPtrRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                        sizeLit(scaleOffset));
      auto dPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, half16PtrType, dPtrRaw)
              .getResult());
      mlir::Value dIndex =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      emitc::SubscriptOp dSubscript =
          rewriter.create<emitc::SubscriptOp>(loc, dPtr, dIndex);
      mlir::Value dHalf =
          rewriter.create<emitc::CastOp>(loc, half16Type, d).getResult();
      rewriter.create<emitc::AssignOp>(loc, dSubscript.getResult(), dHalf);

      // x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);  -- scale every lane by id.
      mlir::Value x0 = vcall(f32m8Type, "__riscv_vfmul_vf_f32m8",
                             mlir::ValueRange{vx, id, vl});

      // The NARROWING CONVERT (ggml riscv/quants.c:60-61): f32 -> i16 (the
      // rounding crux: vfncvt_x_f_w_i16m4 = dynamic frm = round-to-nearest-EVEN),
      // then i16 -> i8 truncate (vncvt). Both callees are ggml's exact spellings.
      mlir::Value vi = vcall(i16m4Type, "__riscv_vfncvt_x_f_w_i16m4",
                             mlir::ValueRange{x0, vl});
      mlir::Value vs = vcall(i8m2Type, "__riscv_vncvt_x_x_w_i8m2",
                             mlir::ValueRange{vi, vl});

      // __riscv_vse8_v_i8m2(yb + 2, vs, vl);  -- store the 32 int8 qs.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qs_store"));
      mlir::Value qsPtrRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                        sizeLit(quantOffset));
      mlir::Value qsPtr =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, qsPtrRaw).getResult();
      emitOpaqueCallVoid(rewriter, loc, "__riscv_vse8_v_i8m2",
                         mlir::ValueRange{qsPtr, vs, vl}, opName, role);
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

    // Per-format AoS block layout facts (ggml-common.h + QK*_0/1 = 32). offsets are
    // byte offsets into the AoS block; qsElem is the qs element ctype (int8 for the
    // bare-scale q8_0, uint8 nibble carrier otherwise). `sub` = the pre-scale bias
    // subtracted from the nibble (8 for q4_0, 16 for q5_0), 0 when a min is added
    // instead. hasMin / hasQh gate the q4_1/q5_1 min and the q5_0/q5_1 5th bit.
    int64_t qk = 32;
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

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type indexType = rewriter.getIndexType();
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
    mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
    mlir::Type i8PtrType = emitc::PointerType::get(constI8Type);
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

      // const int8_t/uint8_t *qs = (const .. *)(xb + qsOff);  (the packed quants).
      mlir::Value qsBaseRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(qsOff));
      mlir::Value qsBase =
          rewriter
              .create<emitc::CastOp>(loc, bareInt8 ? i8PtrType : u8PtrType,
                                     qsBaseRaw)
              .getResult();

      if (bareInt8) {
        // q8_0: for (j = 0; j < 32; ++j) y[j] = qs[j] * d;  (bare signed-int8
        // scale; the load sign-extends).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "q8_scale"));
        auto qFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), sizeLit(qk),
                                                  sizeLit(1),
                                                  /*bodyBuilder=*/nullptr);
        mlir::Value j = qFor.getInductionVar();
        {
          mlir::OpBuilder::InsertionGuard qGuard(rewriter);
          rewriter.setInsertionPointToStart(qFor.getBody());
          // const int8_t *qp = qs + j;  float *yp = yb + j;
          mlir::Value qp =
              rewriter.create<emitc::AddOp>(loc, i8PtrType, qsBase, j);
          mlir::Value qi = loadElemAsInt(qp, constI8Type);
          mlir::Value qf =
              rewriter.create<emitc::CastOp>(loc, floatType, qi).getResult();
          mlir::Value yv =
              rewriter.create<emitc::MulOp>(loc, floatType, qf, d);
          mlir::Value yp =
              rewriter.create<emitc::AddOp>(loc, floatPtrType, yb, j);
          storeF32(yp, yv);
        }
        return mlir::success();
      }

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

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
