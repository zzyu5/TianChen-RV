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

// VariantToEmitCFunc deferred / dequant emit methods: the low-precision and
// deferred-wide dequant bodies + their epilogues and the standalone-dequant
// body. Split out of RVVToEmitC.cpp as a pure code move; the emitted C is
// byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitDequantProductReduceSlice(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::LoadOp lhsLoad, tcrvrvv::LoadOp rhsLoad,
    mlir::Operation *productOp, tcrvrvv::StandaloneReduceOp reduce,
    mlir::Value lhsBuffer, mlir::Value rhsBuffer, mlir::Value sliceOffset,
    mlir::Value accVar, tcrvrvv::VectorType accVecType, mlir::Value loadVL,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Load both i8mf4 sources at (base + offset).
    if (mlir::failed(emitLoad(rewriter, loc, lhsLoad, valueMap, sliceOffset,
                              loadVL)))
      return mlir::failure();
    if (mlir::failed(emitLoad(rewriter, loc, rhsLoad, valueMap, sliceOffset,
                              loadVL)))
      return mlir::failure();
    // The product (plain widening, or the packed-i4 nibble-unpack chain).
    if (auto product = llvm::dyn_cast<tcrvrvv::WideningProductOp>(productOp)) {
      if (mlir::failed(
              emitWideningProduct(rewriter, loc, product, valueMap, loadVL)))
        return mlir::failure();
    } else if (auto packed =
                   llvm::dyn_cast<tcrvrvv::PackedI4NibbleUnpackProductOp>(
                       productOp)) {
      if (mlir::failed(emitPackedI4NibbleUnpackProduct(rewriter, loc, packed,
                                                       valueMap, loadVL)))
        return mlir::failure();
    } else {
      return rewriter.notifyMatchFailure(productOp,
                                         "unsupported dequant product op");
    }
    // Reduce into the running accumulator variable: seed = dot_acc_vec;
    // dot_acc_vec = vwredsum(product, seed, vl).
    mlir::Value product = valueMap.lookup(reduce.getInput());
    if (!product)
      return rewriter.notifyMatchFailure(reduce,
                                         "dequant reduce input unmapped");
    auto srcVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getInput().getType());
    if (!srcVecType)
      return rewriter.notifyMatchFailure(reduce,
                                         "dequant reduce input not a vector");
    std::optional<llvm::StringRef> mnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(reduce,
                                         "unsupported dequant reduce kind");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC || !convertVectorTypeToEmitC(srcVecType))
      return rewriter.notifyMatchFailure(reduce,
                                         "dequant reduce type not convertible");
    mlir::Value seed =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    std::string callee = riscvWideningReductionIntrinsicName(
        *mnemonic, vectorDType(srcVecType), srcVecType.getLmul(),
        vectorDType(accVecType));
    mlir::Value reduced =
        emitOpaqueCall(rewriter, loc, accEmitC, callee,
                       mlir::ValueRange{product, seed, loadVL},
                       reduce.getTCRVEmitCLowerableSourceOpName(),
                       reduce.getTCRVEmitCLowerableSourceRole());
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec",
                           reduce.getTCRVEmitCLowerableSourceOpName(),
                           reduce.getTCRVEmitCLowerableSourceRole()));
    rewriter.create<emitc::AssignOp>(loc, accVar, reduced);
    valueMap[reduce.getResult()] = reduced;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitLowPrecisionDequantBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
    tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
    mlir::Type sizeType, llvm::StringRef setvlCallee,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Walk the scope into ordered product/reduce slices + the dequant epilogue.
    llvm::SmallVector<DequantSlice, 2> slices;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp storeOp;
    llvm::SmallVector<mlir::Operation *, 4> epilogueOps;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        // Group lhs/rhs loads with the product/reduce they feed; the first load
        // of each slice opens a new DequantSlice.
        if (slices.empty() || slices.back().reduce)
          slices.push_back(DequantSlice{});
        if (!slices.back().lhsLoad)
          slices.back().lhsLoad = load;
        else
          slices.back().rhsLoad = load;
      } else if (llvm::isa<tcrvrvv::WideningProductOp,
                           tcrvrvv::PackedI4NibbleUnpackProductOp>(op)) {
        if (slices.empty())
          return rewriter.notifyMatchFailure(scope,
                                             "dequant product before its loads");
        slices.back().productOp = &op;
      } else if (auto reduce = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
        if (slices.empty() || !slices.back().productOp)
          return rewriter.notifyMatchFailure(scope,
                                             "dequant reduce before its product");
        slices.back().reduce = reduce;
      } else if (auto deq = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op)) {
        dequant = deq;
        epilogueOps.push_back(&op);
      } else if (auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        storeOp = store;
        epilogueOps.push_back(&op);
      } else if (llvm::isa<tcrvrvv::SplatOp, tcrvrvv::CompareOp,
                           tcrvrvv::SelectOp>(op)) {
        epilogueOps.push_back(&op);
      } else {
        // Any unexpected op (e.g. a leftover vsetvl_region_marker, the gearbox
        // handoff) means this is not the fully-typed dequant body the routine
        // owns yet -- fall back to legacy so nothing is mislowered.
        return rewriter.notifyMatchFailure(
            &op, "unsupported op in low-precision dequant body");
      }
    }
    if (slices.empty() || !dequant || !storeOp)
      return rewriter.notifyMatchFailure(
          scope, "dequant body missing product/reduce slices or epilogue");
    for (const DequantSlice &slice : slices)
      if (!slice.lhsLoad || !slice.rhsLoad || !slice.productOp || !slice.reduce)
        return rewriter.notifyMatchFailure(scope,
                                           "incomplete dequant product slice");

    // The accumulator vector type (i32 m1) and the acc[0] seed buffer.
    auto accVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(slices.front().reduce.getType());
    if (!accVecType || accVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(scope,
                                         "dequant accumulator not an m1 vector");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC)
      return rewriter.notifyMatchFailure(
          scope, "dequant accumulator type not convertible");
    mlir::Value lhsBuffer = valueMap.lookup(slices.front().lhsLoad.getBuffer());
    mlir::Value rhsBuffer = valueMap.lookup(slices.front().rhsLoad.getBuffer());
    mlir::Value accBuffer =
        valueMap.lookup(slices.front().reduce.getAccumulatorSeed());
    if (!lhsBuffer || !rhsBuffer || !accBuffer)
      return rewriter.notifyMatchFailure(scope,
                                         "dequant body buffers unmapped");

    // unroll_factor (structural op attr): the main loop carries this many copies
    // of the single typed product/reduce slice and steps by vlmax * unroll_factor;
    // absent or 1 == a single plain loop. The body carries exactly ONE typed slice
    // (the template); the conversion expands it `unroll` times. Each expanded copy
    // is byte-identical except for its sliceIndex-derived pointer offset and VL,
    // computed below -- so a single-slice unroll=2 body emits exactly the legacy
    // two-slice grouped C.
    int64_t unroll = 1;
    if (auto u = scope->getAttrOfType<mlir::IntegerAttr>("unroll_factor"))
      unroll = u.getInt();
    if (unroll < 1)
      return rewriter.notifyMatchFailure(
          scope, "dequant unroll_factor must be a positive integer");
    if (slices.size() != 1)
      return rewriter.notifyMatchFailure(
          scope, "dequant body must carry exactly one typed product/reduce slice "
                 "(unrolled via the with_vl unroll_factor attr)");

    llvm::StringRef reduceOpName =
        slices.front().reduce.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef reduceRole =
        slices.front().reduce.getTCRVEmitCLowerableSourceRole();

    // Function-scoped i32 accumulator variable: vint32m1_t dot_acc_vec;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("dot_acc_vec", reduceOpName, reduceRole));
    auto accLValueType = emitc::LValueType::get(accEmitC);
    auto accVar = rewriter.create<emitc::VariableOp>(
        loc, accLValueType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // Optional grouped tail bound: tail_start = (n / (vlmax*2)) * (vlmax*2).
    mlir::Value tailStart;
    if (unroll > 1) {
      mlir::Value unrollLit = rewriter.create<emitc::LiteralOp>(
          loc, sizeType, llvm::Twine(unroll).str());
      mlir::Value step =
          rewriter.create<emitc::MulOp>(loc, sizeType, vlmax, unrollLit);
      mlir::Value chunks =
          rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, step);
      tailStart = rewriter.create<emitc::MulOp>(loc, sizeType, chunks, step);
    }

    // Seed the accumulator from acc[0]: dot_acc_vec = vmv_v_x(acc[0], 1).
    mlir::Value seedSplat = emitScalarSeedSplat(rewriter, loc, accBuffer,
                                                accVecType, reduceOpName,
                                                reduceRole);
    if (!seedSplat)
      return rewriter.notifyMatchFailure(scope,
                                         "dequant accumulator seed not convertible");
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec", reduceOpName, reduceRole));
    rewriter.create<emitc::AssignOp>(loc, accVar, seedSplat);

    // The main chunk loop: for (i = 0; i < <bound>; i += vlmax*unroll).
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value mainBound = tailStart ? tailStart : avlArg;
    mlir::Value mainStep = vlmax;
    if (unroll > 1) {
      mlir::Value unrollLit = rewriter.create<emitc::LiteralOp>(
          loc, sizeType, llvm::Twine(unroll).str());
      mainStep = rewriter.create<emitc::MulOp>(loc, sizeType, vlmax, unrollLit);
    }
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, zero, mainBound, mainStep,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      mlir::Value inductionVar = mainLoop.getInductionVar();
      const DequantSlice &slice = slices.front();
      // Expand the single typed slice `unroll` times. The per-copy pointer offset
      // and VL derive from sliceIndex (not from any slice's own ops), so emitting
      // slices.front() with sliceIndex 0..unroll-1 reproduces the legacy unrolled
      // grouped C byte-for-byte.
      for (int64_t sliceIndex = 0; sliceIndex < unroll; ++sliceIndex) {
        // The slice's runtime VL is setvl(n - i - sliceIndex*vlmax); the second
        // unroll slice loads at (base + i + vlmax) and its VL excludes the first
        // slice's lanes.
        mlir::Value sliceOffset = inductionVar;
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
        if (sliceIndex > 0) {
          // base + i + sliceIndex*vlmax ; remaining -= sliceIndex*vlmax (via vl)
          for (int64_t k = 0; k < sliceIndex; ++k)
            sliceOffset =
                rewriter.create<emitc::AddOp>(loc, sizeType, sliceOffset, vlmax);
          for (int64_t k = 0; k < sliceIndex; ++k)
            remaining =
                rewriter.create<emitc::SubOp>(loc, sizeType, remaining, vlmax);
        }
        mlir::Value sliceVL = emitOpaqueCall(
            rewriter, loc, sizeType, setvlCallee, mlir::ValueRange{remaining},
            preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
            preLoopSetVL.getTCRVEmitCLowerableSourceRole());
        if (mlir::failed(emitDequantProductReduceSlice(
                rewriter, loc, slice.lhsLoad, slice.rhsLoad, slice.productOp,
                slice.reduce, lhsBuffer, rhsBuffer, sliceOffset, accVar,
                accVecType, sliceVL, valueMap)))
          return mlir::failure();
      }
    }

    // The scalar tail loop (only when unrolled): one slice over the remainder,
    // for (i = tail_start; i < n; i += vlmax).
    if (unroll > 1) {
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, tailStart, avlArg,
                                                    vlmax, /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      mlir::Value inductionVar = tailLoop.getInductionVar();
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value sliceVL = emitOpaqueCall(
          rewriter, loc, sizeType, setvlCallee, mlir::ValueRange{remaining},
          preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
          preLoopSetVL.getTCRVEmitCLowerableSourceRole());
      const DequantSlice &slice = slices.front();
      if (mlir::failed(emitDequantProductReduceSlice(
              rewriter, loc, slice.lhsLoad, slice.rhsLoad, slice.productOp,
              slice.reduce, lhsBuffer, rhsBuffer, inductionVar, accVar,
              accVecType, sliceVL, valueMap)))
        return mlir::failure();
    }

    // Dequant epilogue (run once after the loops): scalar extract -> f32 ->
    // *scale -> splat -> store; optionally clamped via splat/compare/select.
    return emitDequantEpilogue(rewriter, loc, variant, dequant, storeOp,
                               epilogueOps, accVar, accVecType, valueMap);
  }

mlir::LogicalResult VariantToEmitCFunc::emitDeferredWideDequantBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
    tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
    mlir::Type sizeType, llvm::StringRef setvlCallee,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Walk the single deferred-wide slice + dequant epilogue.
    tcrvrvv::LoadOp lhsLoad;
    tcrvrvv::LoadOp rhsLoad;
    tcrvrvv::WideningProductOp product;
    tcrvrvv::WideningAccumulateOp accumulate;
    tcrvrvv::StandaloneReduceOp reduce;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp storeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        if (!lhsLoad)
          lhsLoad = load;
        else if (!rhsLoad)
          rhsLoad = load;
        else
          return rewriter.notifyMatchFailure(
              scope, "deferred-wide body carries more than two loads");
      } else if (auto p = llvm::dyn_cast<tcrvrvv::WideningProductOp>(op)) {
        product = p;
      } else if (auto a = llvm::dyn_cast<tcrvrvv::WideningAccumulateOp>(op)) {
        accumulate = a;
      } else if (auto r = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
        reduce = r;
      } else if (auto d = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op)) {
        dequant = d;
      } else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        storeOp = s;
      } else {
        return rewriter.notifyMatchFailure(
            &op, "unsupported op in deferred-wide low-precision dequant body");
      }
    }
    if (!lhsLoad || !rhsLoad || !product || !accumulate || !reduce || !dequant ||
        !storeOp)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide body missing a load/product/accumulate/reduce/"
                 "dequant/store step");

    // The i32m8 deferred accumulator type and the acc[0] seed buffer.
    auto accVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(accumulate.getResult().getType());
    if (!accVecType || accVecType.getLmul() != "m8")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide accumulator not an i32m8 vector");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide accumulator type not convertible");
    auto reduceVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    if (!reduceVecType || reduceVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide trailing reduction result not an m1 vector");
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!reduceEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide trailing reduction type not convertible");

    mlir::Value lhsBuffer = valueMap.lookup(lhsLoad.getBuffer());
    mlir::Value rhsBuffer = valueMap.lookup(rhsLoad.getBuffer());
    mlir::Value accBuffer = valueMap.lookup(reduce.getAccumulatorSeed());
    if (!lhsBuffer || !rhsBuffer || !accBuffer)
      return rewriter.notifyMatchFailure(scope,
                                         "deferred-wide body buffers unmapped");

    llvm::StringRef accOpName =
        accumulate.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef accRole = accumulate.getTCRVEmitCLowerableSourceRole();
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);

    // Function-scoped i32m8 accumulator variable: vint32m8_t dot_acc_vec;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("dot_acc_vec", accOpName, accRole));
    auto accLValueType = emitc::LValueType::get(accEmitC);
    auto accVar = rewriter.create<emitc::VariableOp>(
        loc, accLValueType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // The accumulator's own VLMAX: vlmax_acc = __riscv_vsetvlmax_e32m8();
    std::string accVsetvlmaxCallee =
        riscvVsetvlmaxIntrinsicName(accSEW, accLmul);
    mlir::Value accVlmax = emitOpaqueCall(rewriter, loc, sizeType,
                                          accVsetvlmaxCallee, mlir::ValueRange{},
                                          accOpName, accRole);

    // Zero-seed: dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vlmax_acc);
    std::string accZeroSplatCallee =
        riscvIntrinsicName("vmv_v_x", accSEW, accLmul, accDtype);
    mlir::Value accSeed = emitOpaqueCallBuilt(
        rewriter, loc, accEmitC, accZeroSplatCallee, accOpName, accRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroLit =
              b.create<emitc::LiteralOp>(l, getSizeType(rewriter), "0");
          return {zeroLit, accVlmax};
        });
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec", accOpName, accRole));
    rewriter.create<emitc::AssignOp>(loc, accVar, accSeed);

    // The strip loop: for (i = 0; i < n; i += vlmax).
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      mlir::Value inductionVar = mainLoop.getInductionVar();
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value sliceVL = emitOpaqueCall(
          rewriter, loc, sizeType, setvlCallee, mlir::ValueRange{remaining},
          preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
          preLoopSetVL.getTCRVEmitCLowerableSourceRole());
      // Wide i8m2 loads at base + i, the i16m4 widening product (both reuse the
      // narrow emitters -- they derive <dtype><lmul> from the wide typed
      // vectors, so they emit vle8_v_i8m2 / vwmul_vv_i16m4 unchanged).
      if (mlir::failed(emitLoad(rewriter, loc, lhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(emitLoad(rewriter, loc, rhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(
              emitWideningProduct(rewriter, loc, product, valueMap, sliceVL)))
        return mlir::failure();
      mlir::Value productValue = valueMap.lookup(product.getResult());
      if (!productValue)
        return rewriter.notifyMatchFailure(accumulate,
                                           "deferred-wide product unmapped");
      // DEFERRED accumulate: dot_acc_vec = vwadd_wv_i32m8(dot_acc_vec, p, vl).
      std::string accumulateCallee =
          riscvWideningAccumulateIntrinsicName(accSEW, accLmul, accDtype);
      mlir::Value runningAcc =
          rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
      mlir::Value accumulated = emitOpaqueCall(
          rewriter, loc, accEmitC, accumulateCallee,
          mlir::ValueRange{runningAcc, productValue, sliceVL}, accOpName,
          accRole);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("dot_acc_vec", accOpName, accRole));
      rewriter.create<emitc::AssignOp>(loc, accVar, accumulated);
      valueMap[accumulate.getResult()] = accumulated;
    }

    // Trailing reduction + scalar epilogue: ONE vredsum folds the i32m8
    // accumulator, then acc[0] is added as a SCALAR, then dequant.
    return emitDeferredWideEpilogue(rewriter, loc, variant, reduce, dequant,
                                    storeOp, accVar, accVecType, reduceVecType,
                                    accVlmax, accBuffer, valueMap);
  }

mlir::LogicalResult VariantToEmitCFunc::emitDeferredWideEpilogue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrv::exec::VariantOp variant, tcrvrvv::StandaloneReduceOp reduce,
    tcrvrvv::DequantizeOp dequant, tcrvrvv::StoreOp storeOp, mlir::Value accVar,
    tcrvrvv::VectorType accVecType, tcrvrvv::VectorType reduceVecType,
    mlir::Value accVlmax, mlir::Value accBuffer,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::Value scale = valueMap.lookup(dequant.getScale());
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!scale || !outBuffer)
      return rewriter.notifyMatchFailure(
          dequant, "deferred-wide dequant epilogue operands unmapped");
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequant.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant result not an f32 vector");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!resultEmitC || !accEmitC || !reduceEmitC)
      return rewriter.notifyMatchFailure(
          dequant, "deferred-wide epilogue type not convertible");

    llvm::StringRef reduceOpName =
        reduce.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef reduceRole = reduce.getTCRVEmitCLowerableSourceRole();
    llvm::StringRef opName = dequant.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = dequant.getTCRVEmitCLowerableSourceRole();
    mlir::Type sizeType = getSizeType(rewriter);
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);
    llvm::StringRef reduceDtype = vectorDType(reduceVecType);

    // The reduction destination m1 VLMAX + zero seed:
    //   vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vsetvlmax_e32m1());
    std::string redVsetvlmaxCallee =
        riscvVsetvlmaxIntrinsicName(accSEW, reduceVecType.getLmul());
    mlir::Value reduceVlmax = emitOpaqueCall(
        rewriter, loc, sizeType, redVsetvlmaxCallee, mlir::ValueRange{},
        reduceOpName, reduceRole);
    std::string zeroSeedCallee = riscvIntrinsicName(
        "vmv_v_x", accSEW, reduceVecType.getLmul(), reduceDtype);
    mlir::Value reduceZero = emitOpaqueCallBuilt(
        rewriter, loc, reduceEmitC, zeroSeedCallee, reduceOpName, reduceRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroLit =
              b.create<emitc::LiteralOp>(l, getSizeType(rewriter), "0");
          return {zeroLit, reduceVlmax};
        });

    // ONE trailing vredsum over the i32m8 accumulator:
    //   vint32m1_t vred = __riscv_vredsum_vs_i32m8_i32m1(dot_acc_vec, vzero, vlmax_acc);
    std::optional<llvm::StringRef> reduceMnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!reduceMnemonic)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide trailing reduce kind unsupported");
    std::string reduceCallee = riscvWideningReductionIntrinsicName(
        *reduceMnemonic, accDtype, accLmul, reduceDtype);
    mlir::Value accValue =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    mlir::Value reduced = emitOpaqueCall(
        rewriter, loc, reduceEmitC, reduceCallee,
        mlir::ValueRange{accValue, reduceZero, accVlmax}, reduceOpName,
        reduceRole);
    valueMap[reduce.getResult()] = reduced;

    // Extract lane 0: int32_t reduced_scalar = __riscv_vmv_x_s_i32m1_i32(vred);
    std::string extractCallee =
        riscvScalarExtractIntrinsicName(reduceDtype, reduceVecType.getLmul());
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    mlir::Value reducedScalar =
        emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                       mlir::ValueRange{reduced}, reduceOpName, reduceRole);

    // acc[0] as a SCALAR: int32_t sum = acc[0] + reduced_scalar.
    auto accPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(accBuffer);
    if (!accPointer)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide acc seed buffer not a pointer");
    mlir::Value seedIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp seedSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, accPointer, seedIndex);
    auto seedLValueType =
        llvm::cast<emitc::LValueType>(seedSubscript.getResult().getType());
    mlir::Value seedScalar =
        rewriter
            .create<emitc::LoadOp>(loc, seedLValueType.getValueType(),
                                   seedSubscript.getResult())
            .getResult();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, "scalar_acc0_add"));
    mlir::Value sum =
        rewriter.create<emitc::AddOp>(loc, i32Type, seedScalar, reducedScalar);

    // Dequant: float scaled = (float) sum * scale; out[0] = vse32(vfmv_v_f(scaled, 1)).
    unsigned resSEW = vectorElementWidth(resultVecType);
    llvm::StringRef resLmul = resultVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(resultVecType);
    mlir::Type floatType =
        emitc::OpaqueType::get(rewriter.getContext(), "float");
    mlir::Value asFloat =
        rewriter.create<emitc::CastOp>(loc, floatType, sum).getResult();
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, asFloat, scale);
    std::string splatCallee =
        riscvIntrinsicName("vfmv_v_f", resSEW, resLmul, resDtype);
    mlir::Value one;
    mlir::Value dequantSplat = emitOpaqueCallBuilt(
        rewriter, loc, resultEmitC, splatCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          one = b.create<emitc::LiteralOp>(l, sizeType, "1");
          return {scaled, one};
        });
    valueMap[dequant.getResult()] = dequantSplat;
    std::string storeCallee =
        riscvIntrinsicName("vse", resSEW, resLmul, resDtype);
    emitOpaqueCallVoid(rewriter, loc, storeCallee,
                       mlir::ValueRange{outBuffer, dequantSplat, one},
                       storeOp.getTCRVEmitCLowerableSourceOpName(),
                       storeOp.getTCRVEmitCLowerableSourceRole());
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitDeferredWideDotReduceBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
    tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
    mlir::Type sizeType, llvm::StringRef setvlCallee,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::LoadOp lhsLoad;
    tcrvrvv::LoadOp rhsLoad;
    tcrvrvv::WideningProductOp product;
    tcrvrvv::DeferredAccumulateOp accumulate;
    tcrvrvv::StandaloneReduceOp reduce;
    tcrvrvv::StoreOp storeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        if (!lhsLoad)
          lhsLoad = load;
        else if (!rhsLoad)
          rhsLoad = load;
        else
          return rewriter.notifyMatchFailure(
              scope, "deferred-wide dot-reduce body carries more than two loads");
      } else if (auto p = llvm::dyn_cast<tcrvrvv::WideningProductOp>(op)) {
        product = p;
      } else if (auto a = llvm::dyn_cast<tcrvrvv::DeferredAccumulateOp>(op)) {
        accumulate = a;
      } else if (auto r = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
        reduce = r;
      } else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        storeOp = s;
      } else {
        return rewriter.notifyMatchFailure(
            &op, "unsupported op in deferred-wide dot-reduce body");
      }
    }
    if (!lhsLoad || !rhsLoad || !product || !accumulate || !reduce || !storeOp)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce body missing a load/product/"
                 "accumulate/reduce/store step");

    auto accVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(accumulate.getResult().getType());
    // Accept any budget-legal i32 accumulator LMUL ({m1,m2,m4,m8}): the wide
    // m8 rung at the default budget, a narrower m4/m2/m1 rung at a constrained
    // budget. The downstream intrinsic emission is fully type-driven (it derives
    // <dtype><lmul> from accVecType), so the narrow rung emits the SAME deferred-
    // accumulate algorithm (vadd.vv at accLmul + ONE trailing vredsum) at the
    // narrower width -- the all-compiler LMUL-width ablation.
    if (!accVecType || !accVecType.getElementType().isSignlessInteger(32) ||
        (accVecType.getLmul() != "m1" && accVecType.getLmul() != "m2" &&
         accVecType.getLmul() != "m4" && accVecType.getLmul() != "m8"))
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce accumulator not an i32 m1/m2/m4/m8 "
                 "vector");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce accumulator type not convertible");
    auto reduceVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    if (!reduceVecType || reduceVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce trailing reduction result not m1");
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!reduceEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce trailing reduction not convertible");

    mlir::Value lhsBuffer = valueMap.lookup(lhsLoad.getBuffer());
    mlir::Value rhsBuffer = valueMap.lookup(rhsLoad.getBuffer());
    mlir::Value accBuffer = valueMap.lookup(reduce.getAccumulatorSeed());
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!lhsBuffer || !rhsBuffer || !accBuffer || !outBuffer)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce body buffers unmapped");

    llvm::StringRef accOpName = accumulate.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef accRole = accumulate.getTCRVEmitCLowerableSourceRole();
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);

    // Function-scoped i32m8 accumulator: vint32m8_t dot_acc_vec;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("dot_acc_vec", accOpName, accRole));
    auto accLValueType = emitc::LValueType::get(accEmitC);
    auto accVar = rewriter.create<emitc::VariableOp>(
        loc, accLValueType, emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // The accumulator's own VLMAX: vlmax_acc = __riscv_vsetvlmax_e32m8();
    std::string accVsetvlmaxCallee =
        riscvVsetvlmaxIntrinsicName(accSEW, accLmul);
    mlir::Value accVlmax = emitOpaqueCall(rewriter, loc, sizeType,
                                          accVsetvlmaxCallee, mlir::ValueRange{},
                                          accOpName, accRole);

    // Zero-seed: dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vlmax_acc);
    std::string accZeroSplatCallee =
        riscvIntrinsicName("vmv_v_x", accSEW, accLmul, accDtype);
    mlir::Value accSeed = emitOpaqueCallBuilt(
        rewriter, loc, accEmitC, accZeroSplatCallee, accOpName, accRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroLit =
              b.create<emitc::LiteralOp>(l, getSizeType(rewriter), "0");
          return {zeroLit, accVlmax};
        });
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec", accOpName, accRole));
    rewriter.create<emitc::AssignOp>(loc, accVar, accSeed);

    // The strip loop: for (i = 0; i < n; i += vlmax).
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      mlir::Value inductionVar = mainLoop.getInductionVar();
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value sliceVL = emitOpaqueCall(
          rewriter, loc, sizeType, setvlCallee, mlir::ValueRange{remaining},
          preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
          preLoopSetVL.getTCRVEmitCLowerableSourceRole());
      // Wide i16m4 loads at base + i, the i32m8 SINGLE-step widening product
      // (both reuse the narrow emitters -- they derive <dtype><lmul> from the
      // wide typed vectors, so they emit vle16_v_i16m4 / vwmul_vv_i32m8).
      if (mlir::failed(emitLoad(rewriter, loc, lhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(emitLoad(rewriter, loc, rhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(
              emitWideningProduct(rewriter, loc, product, valueMap, sliceVL)))
        return mlir::failure();
      mlir::Value productValue = valueMap.lookup(product.getResult());
      if (!productValue)
        return rewriter.notifyMatchFailure(accumulate,
                                           "deferred-wide dot-reduce product "
                                           "unmapped");
      // DEFERRED NON-widening accumulate: dot_acc_vec =
      //   __riscv_vadd_vv_i32m8(dot_acc_vec, p, vl). riscvIntrinsicName's
      //   default arm appends the "_vv_<dtype><lmul>" arithmetic form.
      std::string accumulateCallee =
          riscvIntrinsicName("vadd", accSEW, accLmul, accDtype);
      mlir::Value runningAcc =
          rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
      mlir::Value accumulated = emitOpaqueCall(
          rewriter, loc, accEmitC, accumulateCallee,
          mlir::ValueRange{runningAcc, productValue, sliceVL}, accOpName,
          accRole);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("dot_acc_vec", accOpName, accRole));
      rewriter.create<emitc::AssignOp>(loc, accVar, accumulated);
      valueMap[accumulate.getResult()] = accumulated;
    }

    // Trailing reduction + scalar epilogue + i32 lane-0 store.
    return emitDeferredWideDotReduceEpilogue(
        rewriter, loc, reduce, storeOp, accVar, accVecType, reduceVecType,
        accVlmax, accBuffer, outBuffer, valueMap);
  }

mlir::LogicalResult VariantToEmitCFunc::emitDeferredWideDotReduceEpilogue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::StandaloneReduceOp reduce, tcrvrvv::StoreOp storeOp,
    mlir::Value accVar, tcrvrvv::VectorType accVecType,
    tcrvrvv::VectorType reduceVecType, mlir::Value accVlmax,
    mlir::Value accBuffer, mlir::Value outBuffer,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!accEmitC || !reduceEmitC)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide dot-reduce epilogue type not convertible");

    llvm::StringRef reduceOpName = reduce.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef reduceRole = reduce.getTCRVEmitCLowerableSourceRole();
    mlir::Type sizeType = getSizeType(rewriter);
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);
    llvm::StringRef reduceDtype = vectorDType(reduceVecType);

    // The reduction destination m1 VLMAX + zero seed.
    std::string redVsetvlmaxCallee =
        riscvVsetvlmaxIntrinsicName(accSEW, reduceVecType.getLmul());
    mlir::Value reduceVlmax = emitOpaqueCall(
        rewriter, loc, sizeType, redVsetvlmaxCallee, mlir::ValueRange{},
        reduceOpName, reduceRole);
    std::string zeroSeedCallee = riscvIntrinsicName(
        "vmv_v_x", accSEW, reduceVecType.getLmul(), reduceDtype);
    mlir::Value reduceZero = emitOpaqueCallBuilt(
        rewriter, loc, reduceEmitC, zeroSeedCallee, reduceOpName, reduceRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroLit =
              b.create<emitc::LiteralOp>(l, getSizeType(rewriter), "0");
          return {zeroLit, reduceVlmax};
        });

    // ONE trailing vredsum over the i32m8 accumulator.
    std::optional<llvm::StringRef> reduceMnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!reduceMnemonic)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide dot-reduce trailing reduce kind unsupported");
    std::string reduceCallee = riscvWideningReductionIntrinsicName(
        *reduceMnemonic, accDtype, accLmul, reduceDtype);
    mlir::Value accValue =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    mlir::Value reduced = emitOpaqueCall(
        rewriter, loc, reduceEmitC, reduceCallee,
        mlir::ValueRange{accValue, reduceZero, accVlmax}, reduceOpName,
        reduceRole);
    valueMap[reduce.getResult()] = reduced;

    // Extract lane 0: int32_t reduced_scalar = __riscv_vmv_x_s_i32m1_i32(vred);
    std::string extractCallee =
        riscvScalarExtractIntrinsicName(reduceDtype, reduceVecType.getLmul());
    mlir::Type i32Type =
        emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    mlir::Value reducedScalar =
        emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                       mlir::ValueRange{reduced}, reduceOpName, reduceRole);

    // acc[0] as a SCALAR: int32_t sum = acc[0] + reduced_scalar.
    auto accPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(accBuffer);
    if (!accPointer)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide dot-reduce acc seed buffer not a pointer");
    mlir::Value seedIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp seedSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, accPointer, seedIndex);
    auto seedLValueType =
        llvm::cast<emitc::LValueType>(seedSubscript.getResult().getType());
    mlir::Value seedScalar =
        rewriter
            .create<emitc::LoadOp>(loc, seedLValueType.getValueType(),
                                   seedSubscript.getResult())
            .getResult();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, "scalar_acc0_add"));
    mlir::Value sum =
        rewriter.create<emitc::AddOp>(loc, i32Type, seedScalar, reducedScalar);

    // out[0] = vse32(__riscv_vmv_v_x_i32m1(sum, 1)).
    unsigned resSEW = vectorElementWidth(reduceVecType);
    llvm::StringRef resLmul = reduceVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(reduceVecType);
    std::string splatCallee =
        riscvIntrinsicName("vmv_v_x", resSEW, resLmul, resDtype);
    llvm::StringRef storeOpName = storeOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef storeRole = storeOp.getTCRVEmitCLowerableSourceRole();
    mlir::Value one;
    mlir::Value storeSplat = emitOpaqueCallBuilt(
        rewriter, loc, reduceEmitC, splatCallee, storeOpName, storeRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          one = b.create<emitc::LiteralOp>(l, sizeType, "1");
          return {sum, one};
        });
    std::string storeCallee =
        riscvIntrinsicName("vse", resSEW, resLmul, resDtype);
    emitOpaqueCallVoid(rewriter, loc, storeCallee,
                       mlir::ValueRange{outBuffer, storeSplat, one}, storeOpName,
                       storeRole);
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitStandaloneDequantBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg,
    mlir::Value vlmax, mlir::Type sizeType, llvm::StringRef setvlCallee,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Resolve the single load/dequantize/store template ops.
    tcrvrvv::LoadOp load;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp store;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto l = llvm::dyn_cast<tcrvrvv::LoadOp>(op))
        load = l;
      else if (auto d = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op))
        dequant = d;
      else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op))
        store = s;
    }
    if (!load || !dequant || !store)
      return rewriter.notifyMatchFailure(
          scope, "standalone dequant body missing load/dequantize/store");

    mlir::Value scale = valueMap.lookup(dequant.getScale());
    if (!scale)
      return rewriter.notifyMatchFailure(scope,
                                         "standalone dequant scale unmapped");

    // The Gearbox unroll factor (schedule fact carried on the realized scope).
    // The simple two-slice expansion below reproduces the legacy u<unroll> loop;
    // require it present and >= 1 so an un-scheduled body falls back.
    int64_t unroll = 0;
    if (auto u =
            scope->getAttrOfType<mlir::IntegerAttr>("tcrv_rvv.gearbox.unroll"))
      unroll = u.getInt();
    if (unroll < 1)
      return rewriter.notifyMatchFailure(
          scope, "standalone dequant body missing positive "
                 "tcrv_rvv.gearbox.unroll schedule fact");

    // v6 = v5 * unroll (the unrolled loop step). For unroll == 1 the step is
    // plain vlmax (no literal multiply, matching the un-unrolled single loop).
    mlir::Value step = vlmax;
    if (unroll > 1) {
      mlir::Value unrollLit = rewriter.create<emitc::LiteralOp>(
          loc, sizeType, llvm::Twine(unroll).str());
      step = rewriter.create<emitc::MulOp>(loc, sizeType, vlmax, unrollLit);
    }

    // for (size_t i = 0; i < n; i += step) { <unroll slices> }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, step,
                                               /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      mlir::Value inductionVar = forOp.getInductionVar();

      // priorVLSum accumulates the runtime VLs of the prior slices in this
      // iteration: slice 0 sees null (offset 0), slice k>0 sees the sum of
      // vl0..vl(k-1). For u2 this is exactly vl0, matching the legacy golden.
      mlir::Value priorVLSum;
      for (int64_t sliceIndex = 0; sliceIndex < unroll; ++sliceIndex) {
        // Remaining AVL for this slice: recompute (n - i) FRESH, then subtract
        // the prior slices' VLs one at a time (the legacy golden subtracts vl0
        // for the second slice via a dedicated `v16 = v15 - v9`).
        mlir::Value sliceVL = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, setvlCallee,
            preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
            preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining =
                  b.create<emitc::SubOp>(l, sizeType, avlArg, inductionVar);
              if (priorVLSum)
                remaining =
                    b.create<emitc::SubOp>(l, sizeType, remaining, priorVLSum);
              return {remaining};
            });

        // load(base + i [+ priorVLSum]) -> dequant chain -> store(out + i [+ ...])
        if (mlir::failed(emitLoad(rewriter, loc, load, valueMap, inductionVar,
                                  sliceVL, priorVLSum)))
          return mlir::failure();
        mlir::Value source = valueMap.lookup(load.getLoaded());
        if (!source)
          return rewriter.notifyMatchFailure(
              scope, "standalone dequant load result unmapped");
        if (mlir::failed(emitDequantizeChain(rewriter, loc, dequant, source,
                                             scale, valueMap, sliceVL)))
          return mlir::failure();
        if (mlir::failed(emitStore(rewriter, loc, store, valueMap, inductionVar,
                                   sliceVL, priorVLSum)))
          return mlir::failure();

        // Accumulate this slice's VL for the NEXT slice's offset/remaining. The
        // final slice has no successor, so skip the (otherwise dead) add to stay
        // byte-identical to the legacy golden (no trailing `vl0 + vl1`).
        if (sliceIndex + 1 < unroll)
          priorVLSum = priorVLSum ? rewriter.create<emitc::AddOp>(
                                        loc, sizeType, priorVLSum, sliceVL)
                                  : sliceVL;
      }
    }
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitDequantEpilogue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrv::exec::VariantOp variant, tcrvrvv::DequantizeOp dequant,
    tcrvrvv::StoreOp storeOp, llvm::ArrayRef<mlir::Operation *> epilogueOps,
    mlir::Value accVar, tcrvrvv::VectorType accVecType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    bool hasClamp = llvm::any_of(epilogueOps, [](mlir::Operation *op) {
      return llvm::isa<tcrvrvv::SelectOp>(op);
    });
    mlir::Value scale = valueMap.lookup(dequant.getScale());
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!scale || !outBuffer)
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant epilogue operands unmapped");
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequant.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant result not an f32 vector");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!resultEmitC || !accEmitC)
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant epilogue type not convertible");
    llvm::StringRef opName = dequant.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = dequant.getTCRVEmitCLowerableSourceRole();
    unsigned resSEW = vectorElementWidth(resultVecType);
    llvm::StringRef resLmul = resultVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(resultVecType);
    mlir::Type sizeType = getSizeType(rewriter);

    // int32_t dot_acc_scalar = __riscv_vmv_x_s_i32m1_i32(dot_acc_vec);
    mlir::Value accValue =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    std::string extractCallee = riscvScalarExtractIntrinsicName(
        vectorDType(accVecType), accVecType.getLmul());
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    mlir::Value scalar =
        emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                       mlir::ValueRange{accValue}, opName, role);
    // float f = (float) dot_acc_scalar; float scaled = f * scale;
    mlir::Type floatType =
        emitc::OpaqueType::get(rewriter.getContext(), "float");
    mlir::Value asFloat =
        rewriter.create<emitc::CastOp>(loc, floatType, scalar).getResult();
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, asFloat, scale);

    // The scaled f32 lane-0 result splat: vfmv_v_f(scaled, 1). Both the plain
    // dequant store and the clamp's compare inputs consume this lane-0 vector.
    std::string splatCallee =
        riscvIntrinsicName("vfmv_v_f", resSEW, resLmul, resDtype);
    mlir::Value one;
    mlir::Value dequantSplat = emitOpaqueCallBuilt(
        rewriter, loc, resultEmitC, splatCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          one = b.create<emitc::LiteralOp>(l, sizeType, "1");
          return {scaled, one};
        });
    valueMap[dequant.getResult()] = dequantSplat;

    mlir::Value valueToStore = dequantSplat;
    if (hasClamp) {
      // The f32 clamp runs on the lane-0 vector at VL=1: splat lower/upper
      // bounds, compare (vmflt), and select (vmerge) against the dequant result,
      // reusing the already-converted Splat/Compare/Select handlers. The store
      // target is the final selected value.
      for (mlir::Operation *epOp : epilogueOps) {
        if (auto splat = llvm::dyn_cast<tcrvrvv::SplatOp>(epOp)) {
          if (mlir::failed(emitSplat(rewriter, loc, splat, valueMap, one)))
            return mlir::failure();
        } else if (auto compare = llvm::dyn_cast<tcrvrvv::CompareOp>(epOp)) {
          if (mlir::failed(emitCompare(rewriter, loc, compare, valueMap, one)))
            return mlir::failure();
        } else if (auto select = llvm::dyn_cast<tcrvrvv::SelectOp>(epOp)) {
          if (mlir::failed(emitSelect(rewriter, loc, select, valueMap, one)))
            return mlir::failure();
          valueToStore = valueMap.lookup(select.getSelected());
        }
      }
      if (!valueToStore)
        return rewriter.notifyMatchFailure(
            variant, "dequant-clamp select result unmapped");
    }

    // vse32 store the lane-0 result to out base (VL=1).
    std::string storeCallee =
        riscvIntrinsicName("vse", resSEW, resLmul, resDtype);
    emitOpaqueCallVoid(rewriter, loc, storeCallee,
                       mlir::ValueRange{outBuffer, valueToStore, one},
                       storeOp.getTCRVEmitCLowerableSourceOpName(),
                       storeOp.getTCRVEmitCLowerableSourceRole());
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
