#include "TianChenRV/Plugin/Scalar/ScalarBackendEmissionDriver.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Plugin/Scalar/ScalarEmitCRouteProvider.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv {
namespace plugin {
namespace scalar {

namespace {

namespace emitc = ::mlir::emitc;
namespace tcrvemitc = ::tianchenrv::conversion::emitc;

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role,
                               llvm::StringRef opInterface) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << opInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef opInterface, llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << opInterface << " callee=" << callee;
  os.flush();
  return text;
}

std::string kernelStepComment(llvm::StringRef opName, llvm::StringRef role,
                              llvm::StringRef step) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << opName << " role=" << role
     << " step=" << step;
  os.flush();
  return text;
}

/// Lowers a selected `tcrv_scalar.compute_skeleton` boundary into a standalone
/// top-level, pure-scalar EmitC function:
///   #include <stdint.h>
///   int32_t tcrv_scalar_compute_skeleton(int32_t);
///   extern "C" void tcrv_emitc_<kernel>_<variant>(void) {
///     // route_source_op + source_op provenance comments
///     int32_t vN = <scalar_immediate>;
///     int32_t vM = tcrv_scalar_compute_skeleton(vN);
///   }
/// The exported function name is derived from the selected kernel+variant and
/// the emitted constant is the op's `scalar_immediate`, so the emission is
/// operand-driven and carries no __riscv_ intrinsics.
class ScalarComputeSkeletonToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::scalar::ComputeSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      tcrv::scalar::ComputeSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::scalar::ComputeSkeletonOp compute, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = compute.getLoc();

    auto variant =
        compute->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        compute->getAttrOfType<mlir::StringAttr>("source_kernel");
    auto immediate =
        compute->getAttrOfType<mlir::IntegerAttr>("scalar_immediate");
    if (!variant || !sourceKernel || !immediate)
      return rewriter.notifyMatchFailure(
          compute, "compute_skeleton requires selected_variant, source_kernel "
                   "and scalar_immediate attributes");
    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    auto lowerable =
        llvm::dyn_cast<tcrvemitc::TCRVEmitCLowerableOpInterface>(
            compute.getOperation());
    if (!lowerable)
      return rewriter.notifyMatchFailure(
          compute, "tcrv_scalar.compute_skeleton must implement "
                   "TCRVEmitCLowerableOpInterface");
    llvm::StringRef sourceOpName =
        lowerable.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();

    const ScalarEmitCConstructionRoute &route =
        getScalarEmitCConstructionRoute();

    auto module = compute->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(compute, "compute has no module");

    mlir::Type i32 = rewriter.getI32Type();

    // Standalone top-level EmitC module: the standard header, the private
    // callee declaration, then the exported function.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Private callee declaration: int32_t tcrv_scalar_compute_skeleton(int32_t).
    mlir::FunctionType calleeType = rewriter.getFunctionType({i32}, {i32});
    llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
    calleeAttrs.push_back(rewriter.getNamedAttr(
        mlir::SymbolTable::getVisibilityAttrName(),
        rewriter.getStringAttr("private")));
    rewriter.create<emitc::FuncOp>(loc, route.callee, calleeType, calleeAttrs);

    // Exported function: extern "C" void <name>(void).
    mlir::FunctionType functionType =
        rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    // Provenance: route_source_op comment, then the source_op step comment.
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole, route.opInterface));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, route.opInterface,
                         route.callee));

    // Operand-driven body: materialize the op's immediate as a scalar constant
    // and feed it to the portable callee.
    auto constant = rewriter.create<emitc::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(immediate.getInt()));
    llvm::SmallVector<mlir::Value, 1> callOperands{constant.getResult()};
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32}, route.callee,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(compute);
    return mlir::success();
  }
};

/// Lowers a selected `tcrv_scalar.tq2_0_q8_k_vec_dot` boundary into a standalone
/// top-level, pure-scalar EmitC function that IS the ggml tq2_0 x q8_K ternary
/// vec_dot:
///   #include <stdint.h>
///   extern "C" void tcrv_emitc_<kernel>_<variant>(
///       int n, float *s, const uint8_t *vx, const int8_t *vy) {
///     float sumf = 0.0f;
///     size_t nb = (size_t)n / 256;
///     for (size_t ib = 0; ib < nb; ib += 1) {
///       const uint8_t *qs = vx + ib*66;
///       const int8_t  *q8 = vy + ib*292 + 4;
///       int sumi = 0;
///       for (size_t j = 0; j < 64; j += 32)
///         for (size_t l = 0; l < 4; l += 1) {
///           int shift = (int)(l * 2);
///           for (size_t k = 0; k < 32; k += 1) {
///             int w = (((int)qs[j + k] >> shift) & 3) - 1;
///             sumi += (int)q8[j*4 + l*32 + k] * w;
///           }
///         }
///       float dy = *(const float *)(vy + ib*292 + 0);
///       float dx = (float)*(const _Float16 *)(vx + ib*66 + 64);
///       sumf = sumf + (float)sumi * (dy * dx);
///     }
///     *s = sumf;
///   }
/// The exported function name is derived from source_kernel + selected_variant,
/// and the block-format facts (qk, strides, byte offsets) drive the emitted
/// loop bounds and address arithmetic, so the emission is operand-driven. The
/// ternary decode is a pure int8xint8 MAC -- NO XOR-popcount, NO __riscv_
/// intrinsics, NO vector machinery.
class ScalarTernaryQ2Q8BlockDotToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::scalar::TernaryQ2Q8BlockDotOp> {
public:
  using mlir::OpConversionPattern<
      tcrv::scalar::TernaryQ2Q8BlockDotOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::scalar::TernaryQ2Q8BlockDotOp dot, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = dot.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    auto variant =
        dot->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel = dot->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          dot, "tq2_0_q8_k_vec_dot requires selected_variant and source_kernel "
               "attributes");

    auto lowerable =
        llvm::dyn_cast<tcrvemitc::TCRVEmitCLowerableOpInterface>(
            dot.getOperation());
    if (!lowerable)
      return rewriter.notifyMatchFailure(
          dot, "tq2_0_q8_k_vec_dot must implement TCRVEmitCLowerableOpInterface");
    llvm::StringRef opName = lowerable.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = lowerable.getTCRVEmitCLowerableSourceRole();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = dot.getQk();                                  // 256
    int64_t weightStride = dot.getWeightBlockStride();         // 66
    int64_t activationStride = dot.getActivationBlockStride(); // 292
    int64_t weightDOffset = dot.getWeightDByteOffset();        // 64
    int64_t activationDOffset = dot.getActivationDByteOffset();//  0
    int64_t q8Offset = dot.getActivationQuantByteOffset();     //  4
    // Fixed 2-bit ternary structure: four ternary weights (planes) packed per
    // byte at shifts {0,2,4,6}, ggml's 32-lane inner strip -> qk/4 packed
    // weight bytes per super-block, walked in 32-byte plane groups.
    int64_t qsBytes = qk / 4;         // 64 packed weight bytes at QK_K=256
    int64_t numPlanes = 4;            // 2-bit fields per byte
    int64_t planeLanes = 32;          // ggml's fixed 32-lane sub-group
    int64_t chunkBytes = planeLanes;  // qs plane-group stride (j += 32)

    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    auto module = dot->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(dot, "op has no module");

    mlir::Type sizeType = emitc::OpaqueType::get(ctx, "size_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constU8PtrType = emitc::PointerType::get(constU8Type);
    mlir::Type constI8PtrType = emitc::PointerType::get(constI8Type);
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    mlir::Type floatPtrType = emitc::PointerType::get(floatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // #include <stdint.h> at module start.
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // extern "C" void <name>(int n, float *s, const uint8_t *vx,
    //                        const int8_t *vy).
    mlir::FunctionType functionType = rewriter.getFunctionType(
        {intType, floatPtrType, constU8PtrType, constI8PtrType}, {});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    mlir::Value nArg = entry->getArgument(0);
    auto sArg = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
        entry->getArgument(1));
    mlir::Value vxArg = entry->getArgument(2);
    mlir::Value vyArg = entry->getArgument(3);

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto intLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, intType, std::to_string(v));
    };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, kernelStepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(opName, role, "TCRVEmitCLowerableOpInterface"));

    // size_t nb = (size_t)n / qk;
    step("super_block_count");
    mlir::Value nSize =
        rewriter.create<emitc::CastOp>(loc, sizeType, nArg).getResult();
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, nSize, sizeLit(qk))
            .getResult();

    // float sumf = 0.0f;
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f").getResult());

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }
    step("super_block_loop");
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // const uint8_t *qs = vx + ib*weightStride;  (weight qs at byte 0).
      mlir::Value xoff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(weightStride))
              .getResult();
      mlir::Value xb =
          rewriter.create<emitc::AddOp>(loc, vxArg.getType(), vxArg, xoff)
              .getResult();
      auto qs = llvm::cast<mlir::TypedValue<emitc::PointerType>>(xb);

      // const int8_t *q8 = vy + ib*activationStride + q8Offset;
      mlir::Value yoff =
          rewriter
              .create<emitc::MulOp>(loc, sizeType, ib, sizeLit(activationStride))
              .getResult();
      mlir::Value yb =
          rewriter.create<emitc::AddOp>(loc, vyArg.getType(), vyArg, yoff)
              .getResult();
      mlir::Value q8base = yb;
      if (q8Offset != 0)
        q8base = rewriter
                     .create<emitc::AddOp>(loc, vyArg.getType(), yb,
                                           sizeLit(q8Offset))
                     .getResult();
      auto q8 = llvm::cast<mlir::TypedValue<emitc::PointerType>>(q8base);

      // int sumi = 0;
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(intType), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar,
          rewriter.create<emitc::LiteralOp>(loc, intType, "0").getResult());

      // for (size_t j = 0; j < qsBytes; j += chunkBytes)  -- 32-byte plane groups.
      step("plane_group_loop");
      auto jLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                 sizeLit(qsBytes),
                                                 sizeLit(chunkBytes),
                                                 /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard jg(rewriter);
        rewriter.setInsertionPointToStart(jLoop.getBody());
        mlir::Value j = jLoop.getInductionVar();

        // for (size_t l = 0; l < numPlanes; l += 1)  -- the 4 2-bit planes.
        auto lLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                   sizeLit(numPlanes),
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
        {
          mlir::OpBuilder::InsertionGuard lg(rewriter);
          rewriter.setInsertionPointToStart(lLoop.getBody());
          mlir::Value l = lLoop.getInductionVar();

          // int shift = (int)(l * 2);  (the 2-bit field shift {0,2,4,6}).
          mlir::Value shift =
              rewriter
                  .create<emitc::CastOp>(
                      loc, intType,
                      rewriter
                          .create<emitc::MulOp>(loc, sizeType, l, sizeLit(2))
                          .getResult())
                  .getResult();

          // for (size_t k = 0; k < planeLanes; k += 1)
          auto kLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                     sizeLit(planeLanes),
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
          {
            mlir::OpBuilder::InsertionGuard kg(rewriter);
            rewriter.setInsertionPointToStart(kLoop.getBody());
            mlir::Value k = kLoop.getInductionVar();

            step("ternary_decode_mac");
            // int w = (((int)qs[j + k] >> shift) & 3) - 1;  (2-bit ternary,
            // {-1,0,+1(,2)}).
            mlir::Value wIdx =
                rewriter.create<emitc::AddOp>(loc, sizeType, j, k).getResult();
            mlir::Value qsElem =
                rewriter.create<emitc::SubscriptOp>(loc, qs, wIdx).getResult();
            mlir::Value qsByte =
                rewriter.create<emitc::LoadOp>(loc, constU8Type, qsElem)
                    .getResult();
            mlir::Value qsInt =
                rewriter.create<emitc::CastOp>(loc, intType, qsByte).getResult();
            mlir::Value shifted =
                rewriter
                    .create<emitc::BitwiseRightShiftOp>(loc, intType, qsInt,
                                                        shift)
                    .getResult();
            mlir::Value masked =
                rewriter
                    .create<emitc::BitwiseAndOp>(loc, intType, shifted,
                                                 intLit(3))
                    .getResult();
            mlir::Value w =
                rewriter.create<emitc::SubOp>(loc, intType, masked, intLit(1))
                    .getResult();

            // int a = (int)q8[j*4 + l*32 + k];  (the matching int8 activation).
            mlir::Value j4 =
                rewriter.create<emitc::MulOp>(loc, sizeType, j, sizeLit(4))
                    .getResult();
            mlir::Value l32 =
                rewriter
                    .create<emitc::MulOp>(loc, sizeType, l, sizeLit(planeLanes))
                    .getResult();
            mlir::Value q8Idx =
                rewriter
                    .create<emitc::AddOp>(
                        loc, sizeType,
                        rewriter.create<emitc::AddOp>(loc, sizeType, j4, l32)
                            .getResult(),
                        k)
                    .getResult();
            mlir::Value q8Elem =
                rewriter.create<emitc::SubscriptOp>(loc, q8, q8Idx).getResult();
            mlir::Value q8Byte =
                rewriter.create<emitc::LoadOp>(loc, constI8Type, q8Elem)
                    .getResult();
            mlir::Value q8Int =
                rewriter.create<emitc::CastOp>(loc, intType, q8Byte)
                    .getResult();

            // sumi += a * w;  (integer int8xint8 MAC; order-free).
            mlir::Value prod =
                rewriter.create<emitc::MulOp>(loc, intType, q8Int, w)
                    .getResult();
            mlir::Value sumiCur =
                rewriter.create<emitc::LoadOp>(loc, intType, sumiVar)
                    .getResult();
            mlir::Value sumiNext =
                rewriter.create<emitc::AddOp>(loc, intType, sumiCur, prod)
                    .getResult();
            rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
          }
        }
      }

      // float dy = *(const float *)(vy + ib*activationStride + activationDOffset).
      step("fold_activation_d");
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter
                     .create<emitc::AddOp>(loc, vyArg.getType(), yb,
                                           sizeLit(activationDOffset))
                     .getResult();
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex)
              .getResult();
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();

      // float dx = (float)*(const _Float16 *)(vx + ib*weightStride +
      // weightDOffset).
      step("fold_weight_d");
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter
                     .create<emitc::AddOp>(loc, vxArg.getType(), xb,
                                           sizeLit(weightDOffset))
                     .getResult();
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dxAddr})
              .getResult(0);

      // float d = dy * dx;  (the single per-super-block scale).
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dy, dx).getResult();

      // sumf = sumf + (float)sumi * d;  -- ONE emitc.expression so it renders as
      // ggml's single `sumf += (float) sumi * d;` C statement.
      step("scalar_fold");
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, intType, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto foldExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard eg(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&foldExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiF =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value term =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiF, d).getResult();
        mlir::Value next =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, term)
                .getResult();
        rewriter.create<emitc::YieldOp>(loc, next);
      }
      rewriter.create<emitc::AssignOp>(loc, sumfVar, foldExpr.getResult());
    }

    // *s = sumf;  (structured scalar store through the float * output pointer).
    step("store_s");
    mlir::Value sumf =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSub =
        rewriter.create<emitc::SubscriptOp>(loc, sArg, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSub.getResult(), sumf);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(dot);
    return mlir::success();
  }
};

/// Lowers a selected `tcrv_scalar.dequantize_row_q4_0` boundary into a standalone
/// top-level, pure-scalar EmitC function that IS the ggml q4_0 dequantize_row:
///   #include <stdint.h>
///   extern "C" void tcrv_emitc_<kernel>_<variant>(
///       int n, float *y, const uint8_t *vx) {
///     size_t nb = (size_t)n / 32;
///     for (size_t ib = 0; ib < nb; ib += 1) {
///       const uint8_t *xb = vx + ib*18;
///       float d = (float)*(const _Float16 *)(xb + 0);
///       const uint8_t *qs = xb + 2;
///       size_t yb = ib * 32;
///       for (size_t j = 0; j < 16; j += 1) {
///         const uint8_t q = qs[j];
///         int x0 = ((int)q & 15) - 8;
///         int x1 = ((int)q >> 4) - 8;
///         y[yb + j]      = (float)x0 * d;
///         y[yb + j + 16] = (float)x1 * d;
///       }
///     }
///   }
/// The exported function name is derived from source_kernel + selected_variant,
/// and the block-format facts (qk, stride, byte offsets) drive the emitted loop
/// bounds and address arithmetic, so the emission is operand-driven. The nibble
/// decode is pure integer arithmetic -- NO XOR-popcount, NO __riscv_ intrinsics,
/// NO vector machinery.
class ScalarDequantizeRowQ4ToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::scalar::DequantizeRowQ4Op> {
public:
  using mlir::OpConversionPattern<
      tcrv::scalar::DequantizeRowQ4Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::scalar::DequantizeRowQ4Op dequant, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = dequant.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    auto variant =
        dequant->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        dequant->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          dequant, "dequantize_row_q4_0 requires selected_variant and "
                   "source_kernel attributes");

    auto lowerable =
        llvm::dyn_cast<tcrvemitc::TCRVEmitCLowerableOpInterface>(
            dequant.getOperation());
    if (!lowerable)
      return rewriter.notifyMatchFailure(
          dequant,
          "dequantize_row_q4_0 must implement TCRVEmitCLowerableOpInterface");
    llvm::StringRef opName = lowerable.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = lowerable.getTCRVEmitCLowerableSourceRole();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = dequant.getQk();                              // 32
    int64_t weightStride = dequant.getWeightBlockStride();     // 18
    int64_t weightDOffset = dequant.getWeightDByteOffset();    //  0
    int64_t weightQuantOffset = dequant.getWeightQuantByteOffset(); // 2
    // q4_0 packs two 4-bit weights per byte: the low nibble expands into the
    // first half of the block, the high nibble into the second half.
    int64_t half = qk / 2; // 16 packed weight bytes / output half width

    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    auto module = dequant->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(dequant, "op has no module");

    mlir::Type sizeType = emitc::OpaqueType::get(ctx, "size_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constU8PtrType = emitc::PointerType::get(constU8Type);
    mlir::Type floatPtrType = emitc::PointerType::get(floatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // #include <stdint.h> at module start.
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // extern "C" void <name>(int n, float *y, const uint8_t *vx).
    mlir::FunctionType functionType =
        rewriter.getFunctionType({intType, floatPtrType, constU8PtrType}, {});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    mlir::Value nArg = entry->getArgument(0);
    auto yArg = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
        entry->getArgument(1));
    mlir::Value vxArg = entry->getArgument(2);

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto intLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, intType, std::to_string(v));
    };
    auto step = [&](llvm::StringRef s) {
      rewriter.create<emitc::VerbatimOp>(loc, kernelStepComment(opName, role, s));
    };

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(opName, role, "TCRVEmitCLowerableOpInterface"));

    // size_t nb = (size_t)n / qk;
    step("block_count");
    mlir::Value nSize =
        rewriter.create<emitc::CastOp>(loc, sizeType, nArg).getResult();
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, nSize, sizeLit(qk))
            .getResult();

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }
    step("block_loop");
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // const uint8_t *xb = vx + ib*weightStride;
      mlir::Value xoff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(weightStride))
              .getResult();
      mlir::Value xb =
          rewriter.create<emitc::AddOp>(loc, vxArg.getType(), vxArg, xoff)
              .getResult();

      // float d = (float)*(const _Float16 *)(xb + weightDOffset);
      step("block_scale");
      mlir::Value dAddr = xb;
      if (weightDOffset != 0)
        dAddr = rewriter
                    .create<emitc::AddOp>(loc, vxArg.getType(), xb,
                                          sizeLit(weightDOffset))
                    .getResult();
      mlir::Value d =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dAddr})
              .getResult(0);

      // const uint8_t *qs = xb + weightQuantOffset;
      mlir::Value qsBase = xb;
      if (weightQuantOffset != 0)
        qsBase = rewriter
                     .create<emitc::AddOp>(loc, vxArg.getType(), xb,
                                           sizeLit(weightQuantOffset))
                     .getResult();
      auto qs = llvm::cast<mlir::TypedValue<emitc::PointerType>>(qsBase);

      // size_t yb = ib * qk;  (the block's base index in the float output row).
      mlir::Value yb =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk))
              .getResult();

      // for (size_t j = 0; j < qk/2; j += 1)
      step("nibble_loop");
      auto jLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), sizeLit(half),
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard jg(rewriter);
        rewriter.setInsertionPointToStart(jLoop.getBody());
        mlir::Value j = jLoop.getInductionVar();

        step("nibble_decode");
        // const uint8_t q = qs[j];  int qi = (int)q;
        mlir::Value qsElem =
            rewriter.create<emitc::SubscriptOp>(loc, qs, j).getResult();
        mlir::Value qByte =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, qsElem).getResult();
        mlir::Value qInt =
            rewriter.create<emitc::CastOp>(loc, intType, qByte).getResult();

        // int x0 = (qi & 15) - 8;   (low nibble, zero-centered to [-8, 7]).
        mlir::Value lo =
            rewriter.create<emitc::BitwiseAndOp>(loc, intType, qInt, intLit(15))
                .getResult();
        mlir::Value x0 =
            rewriter.create<emitc::SubOp>(loc, intType, lo, intLit(8))
                .getResult();

        // int x1 = (qi >> 4) - 8;   (high nibble).
        mlir::Value hi =
            rewriter
                .create<emitc::BitwiseRightShiftOp>(loc, intType, qInt,
                                                    intLit(4))
                .getResult();
        mlir::Value x1 =
            rewriter.create<emitc::SubOp>(loc, intType, hi, intLit(8))
                .getResult();

        // y[yb + j] = (float)x0 * d;  (low nibble scatters to the first half).
        step("scatter_low");
        mlir::Value o0 =
            rewriter.create<emitc::AddOp>(loc, sizeType, yb, j).getResult();
        mlir::Value x0f =
            rewriter.create<emitc::CastOp>(loc, floatType, x0).getResult();
        mlir::Value p0 =
            rewriter.create<emitc::MulOp>(loc, floatType, x0f, d).getResult();
        emitc::SubscriptOp out0 =
            rewriter.create<emitc::SubscriptOp>(loc, yArg, o0);
        rewriter.create<emitc::AssignOp>(loc, out0.getResult(), p0);

        // y[yb + j + qk/2] = (float)x1 * d;  (high nibble -> the second half).
        step("scatter_high");
        mlir::Value o1 =
            rewriter.create<emitc::AddOp>(loc, sizeType, o0, sizeLit(half))
                .getResult();
        mlir::Value x1f =
            rewriter.create<emitc::CastOp>(loc, floatType, x1).getResult();
        mlir::Value p1 =
            rewriter.create<emitc::MulOp>(loc, floatType, x1f, d).getResult();
        emitc::SubscriptOp out1 =
            rewriter.create<emitc::SubscriptOp>(loc, yArg, o1);
        rewriter.create<emitc::AssignOp>(loc, out1.getResult(), p1);
      }
    }

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(dequant);
    return mlir::success();
  }
};

class ScalarBackendEmissionDriver final
    : public tcrvemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "scalar"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {
    // The scalar skeleton carries no scalar-typed dataflow values; the identity
    // conversion installed by the harness suffices.
  }

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<tcrv::scalar::ComputeSkeletonOp,
                        tcrv::scalar::TernaryQ2Q8BlockDotOp,
                        tcrv::scalar::DequantizeRowQ4Op>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<ScalarComputeSkeletonToEmitCFunc,
                 ScalarTernaryQ2Q8BlockDotToEmitCFunc,
                 ScalarDequantizeRowQ4ToEmitCFunc>(typeConverter,
                                                   patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasScalar = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          tcrv::scalar::TCRVScalarDialect::getDialectNamespace()) {
        hasScalar = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasScalar;
  }
};

llvm::LogicalResult
ScalarBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
  // Once a function was produced, drop the now-emptied tcrv.exec scaffolding
  // (kernel/capability/diagnostics) and any leftover source ops so the module
  // is the clean, standalone EmitC-only shape the emitc->C++ emitter expects.
  bool producedFunc = false;
  module.walk([&](emitc::FuncOp) { producedFunc = true; });
  if (!producedFunc)
    return llvm::success();

  llvm::SmallVector<mlir::Operation *, 2> drainedTopLevel;
  for (mlir::Operation &op : module.getBody()->getOperations()) {
    llvm::StringRef dialect = op.getName().getDialectNamespace();
    if (dialect != emitc::EmitCDialect::getDialectNamespace())
      drainedTopLevel.push_back(&op);
  }
  for (mlir::Operation *op : drainedTopLevel)
    op->erase();
  return llvm::success();
}

} // namespace

void registerScalarBackendEmitter(
    tcrvemitc::BackendEmissionRegistry &registry) {
  // Function-local static: owned by this translation unit, outlives the
  // registry, no global-init-order hazard.
  static const ScalarBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace scalar
} // namespace plugin
} // namespace tianchenrv
