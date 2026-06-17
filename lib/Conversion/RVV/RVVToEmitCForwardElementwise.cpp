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

mlir::LogicalResult VariantToEmitCFunc::emitGgmlVecScaleF32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlVecScaleF32Op scaleOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto s = llvm::dyn_cast<tcrvrvv::GgmlVecScaleF32Op>(op))
        scaleOp = s;
    }
    if (!scaleOp)
      return rewriter.notifyMatchFailure(scope, "scale body missing the op");

    mlir::Value buffer = valueMap.lookup(scaleOp.getBuffer());
    mlir::Value scalar = valueMap.lookup(scaleOp.getScalar());
    if (!buffer || !scalar)
      return rewriter.notifyMatchFailure(scaleOp,
                                         "scale ABI operand unmapped");

    llvm::StringRef opName = scaleOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = scaleOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type bufferPtrType = buffer.getType();

    // The f32 strip-loop LMUL is a bounded resource/scheduling fact (default m8,
    // matching ggml's hand-written path). It is the *how* (vector grouping /
    // strip width), never the *what*: the result is byte-exact at any anchor
    // (every lane is multiplied by the same scalar v). The verifier bounds it to
    // m1|m2|m4|m8 (carried as "strip_lmul" so the op holds no forbidden
    // dataflow-parameter "lmul" at the I5 boundary).
    llvm::StringRef lmul = scaleOp.getStripLmul().value_or("m8");
    std::string f32VecTypeName = ("vfloat32" + lmul + "_t").str();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, f32VecTypeName);
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m<L>(n).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, setvlCallee));
    mlir::Value vlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         setvlCallee, mlir::ValueRange{avlArg})
            .getResult(0);

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // Remaining-AVL setvl: size_t vl = __riscv_vsetvl_e32m<L>(n - i).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, setvlCallee));
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);

      // In-place element pointer: float *p = y + i.
      mlir::Value elemPtr = rewriter.create<emitc::AddOp>(
          loc, bufferPtrType, buffer, inductionVar);
      mlir::Value loadPtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, elemPtr).getResult();

      // vfloat32m<L>_t ay = __riscv_vle32_v_f32m<L>(p, vl);
      std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      mlir::Value ay =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           loadCallee,
                                           mlir::ValueRange{loadPtr, bodyVL})
              .getResult(0);

      // vfloat32m<L>_t ny = __riscv_vfmul_vf_f32m<L>(ay, v, vl);  scalar bcast.
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value ny =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           mulCallee,
                                           mlir::ValueRange{ay, scalar, bodyVL})
              .getResult(0);

      // __riscv_vse32_v_f32m<L>(p, ny, vl);  store back in place.
      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, storeCallee));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, storeCallee,
          mlir::ValueRange{loadPtr, ny, bodyVL});
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlRmsNormF32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRmsNormF32Op rmsOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto r = llvm::dyn_cast<tcrvrvv::GgmlRmsNormF32Op>(op))
        rmsOp = r;
    }
    if (!rmsOp)
      return rewriter.notifyMatchFailure(scope, "rms_norm body missing the op");

    mlir::Value input = valueMap.lookup(rmsOp.getInput());
    mlir::Value outputBuf = valueMap.lookup(rmsOp.getOutput());
    mlir::Value eps = valueMap.lookup(rmsOp.getEps());
    if (!input || !outputBuf || !eps)
      return rewriter.notifyMatchFailure(rmsOp,
                                         "rms_norm ABI operand unmapped");

    llvm::StringRef opName = rmsOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = rmsOp.getTCRVEmitCLowerableSourceRole();
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
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "scale"));
    mlir::Value meanPlusEps =
        rewriter.create<emitc::AddOp>(loc, floatType, mean, eps);
    mlir::Value sqrtVal =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                         "sqrtf",
                                         mlir::ValueRange{meanPlusEps})
            .getResult(0);
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
    llvm::StringRef lmul = rmsOp.getStripLmul().value_or("m8");
    std::string f32VecTypeName = ("vfloat32" + lmul + "_t").str();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, f32VecTypeName);

    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, setvlCallee));
    mlir::Value vlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         setvlCallee, mlir::ValueRange{avlArg})
            .getResult(0);

    mlir::Value normZero =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto normFor = rewriter.create<emitc::ForOp>(loc, normZero, avlArg, vlmax,
                                                 /*bodyBuilder=*/nullptr);
    mlir::Value normIdx = normFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(normFor.getBody());

      // size_t vl = __riscv_vsetvl_e32m<L>(ne00 - i);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, setvlCallee));
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, normIdx);
      mlir::Value bodyVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);

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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      mlir::Value vx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           loadCallee,
                                           mlir::ValueRange{xLoadPtr, bodyVL})
              .getResult(0);

      // vfloat32m<L>_t vy = __riscv_vfmul_vf_f32m<L>(vx, scale, vl);
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value vy =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           mulCallee,
                                           mlir::ValueRange{vx, scale, bodyVL})
              .getResult(0);

      // __riscv_vse32_v_f32m<L>(yp, vy, vl);
      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, storeCallee));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, storeCallee,
          mlir::ValueRange{yStorePtr, vy, bodyVL});
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
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resTy}, callee, args)
          .getResult(0);
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

mlir::LogicalResult VariantToEmitCFunc::emitGgmlVecSiluF32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlVecSiluF32Op siluOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto s = llvm::dyn_cast<tcrvrvv::GgmlVecSiluF32Op>(op))
        siluOp = s;
    }
    if (!siluOp)
      return rewriter.notifyMatchFailure(scope, "silu body missing the op");

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
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resTy}, callee, args)
          .getResult(0);
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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "__riscv_vse32_v_f32m2"));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, "__riscv_vse32_v_f32m2",
          mlir::ValueRange{yStorePtr, vy, bodyVL});
    }

    return mlir::success();
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitGgmlVecSoftMaxF32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlVecSoftMaxF32Op softMaxOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto s = llvm::dyn_cast<tcrvrvv::GgmlVecSoftMaxF32Op>(op))
        softMaxOp = s;
    }
    if (!softMaxOp)
      return rewriter.notifyMatchFailure(scope, "soft_max body missing the op");

    mlir::Value outputBuf = valueMap.lookup(softMaxOp.getOutput());
    mlir::Value input = valueMap.lookup(softMaxOp.getInput());
    mlir::Value maxArg = valueMap.lookup(softMaxOp.getMax());
    if (!outputBuf || !input || !maxArg)
      return rewriter.notifyMatchFailure(softMaxOp,
                                         "soft_max ABI operand unmapped");

    llvm::StringRef opName = softMaxOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = softMaxOp.getTCRVEmitCLowerableSourceRole();
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
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resTy}, callee, args)
          .getResult(0);
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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "__riscv_vse32_v_f32m2"));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, "__riscv_vse32_v_f32m2",
          mlir::ValueRange{yStorePtr, val, bodyVL});

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
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultType}, callee,
                                       args)
          .getResult(0);
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
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "__riscv_vse8_v_i8m2"));
      rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{},
                                           "__riscv_vse8_v_i8m2",
                                           mlir::ValueRange{qsPtr, vs, vl});
    }

    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitGgmlRopeNormF32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRopeNormF32Op ropeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto r = llvm::dyn_cast<tcrvrvv::GgmlRopeNormF32Op>(op))
        ropeOp = r;
    }
    if (!ropeOp)
      return rewriter.notifyMatchFailure(scope, "rope body missing the op");

    mlir::Value input = valueMap.lookup(ropeOp.getInput());
    mlir::Value output = valueMap.lookup(ropeOp.getOutput());
    mlir::Value thetaBase = valueMap.lookup(ropeOp.getThetaBase());
    mlir::Value thetaScale = valueMap.lookup(ropeOp.getThetaScale());
    if (!input || !output || !thetaBase || !thetaScale)
      return rewriter.notifyMatchFailure(ropeOp, "rope ABI operand unmapped");

    llvm::StringRef opName = ropeOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = ropeOp.getTCRVEmitCLowerableSourceRole();
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
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "cosf"));
      mlir::Value cosT =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           "cosf", mlir::ValueRange{thetaCur})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "sinf"));
      mlir::Value sinT =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           "sinf", mlir::ValueRange{thetaCur})
              .getResult(0);

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

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
