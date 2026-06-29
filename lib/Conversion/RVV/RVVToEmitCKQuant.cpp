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

// VariantToEmitCFunc K-quant (super-block) emit methods: q2_K/q3_K/q4_K/q5_K/
// q6_K block-dots plus the shared aux32 integer-core / partial helpers. Split
// out of RVVToEmitC.cpp as a pure code move; the emitted C is byte-identical.

mlir::TypedValue<emitc::LValueType> VariantToEmitCFunc::emitQ6_KSuperBlockAux32Core(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q6_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    // base + fixed byte offset, cast to a typed (const uint8_t* / const int8_t*).
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };
    // u8 immediate vector op: __riscv_<mnemonic>_<dtype><lmul>(src, imm, vl).
    // This idiom interleaves the immediate LiteralOp BETWEEN the step-comment
    // VerbatimOp and the call (emitc-dialect dump order [verbatim, literal,
    // call]), so it routes through emitVCallBuilt (the L0b interleave variant):
    // the immediate is built INSIDE the buildOperands callback, after the
    // verbatim, preserving the order byte-for-byte.
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef imm, mlir::Value vl) -> mlir::Value {
      return emitVCallBuilt(
          rewriter, loc, cx.u8m2Type, mnemonic, "u8m2", opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, cx.i32ImmType, imm.str());
            return {src, amt, vl};
          });
    };
    auto u8VVOp = [&](llvm::StringRef mnemonic, mlir::Value a, mlir::Value b,
                      mlir::Value vl) -> mlir::Value {
      return emitVCall(rewriter, loc, cx.u8m2Type, mnemonic, "u8m2",
                       mlir::ValueRange{a, b, vl}, opName, role);
    };
    auto u8Load = [&](mlir::Value ptr, mlir::Value vl) -> mlir::Value {
      return emitVCall(rewriter, loc, cx.u8m2Type, "vle8_v", "u8m2",
                       mlir::ValueRange{ptr, vl}, opName, role);
    };

    // ---- (A) 6-bit ql+qh unpack into aux8 (element-ordered, biased -32) ----
    // Two 128-element chunks; per chunk a 32-wide e8m2 strip computes the four
    // element groups a[c+0/+32/+64/+96] and stores each into aux8.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "unpack_6bit"));
    std::string unpackSetvl = "__riscv_vsetvl_e8m2";
    for (int64_t chunk = 0; chunk < 2; ++chunk) {
      int64_t qlChunk = chunk * 64; // ql advances 64 bytes per 128-elem chunk
      int64_t qhChunk = chunk * 32; // qh advances 32 bytes per 128-elem chunk
      int64_t aChunk = chunk * 128; // aux8 element base for this chunk
      mlir::Value vlu = emitOpaqueCallBuilt(
          rewriter, loc, cx.sizeType, unpackSetvl, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(32)};
          });
      mlir::Value ql0Ptr =
          byteOffsetPtr(xb, cx.weightPtrType, qlChunk + 0, cx.u8PtrType);
      mlir::Value ql1Ptr =
          byteOffsetPtr(xb, cx.weightPtrType, qlChunk + 32, cx.u8PtrType);
      mlir::Value qhPtr =
          byteOffsetPtr(xb, cx.weightPtrType, cx.qhOffset + qhChunk,
                        cx.u8PtrType);
      mlir::Value ql0 = u8Load(ql0Ptr, vlu);
      mlir::Value ql1 = u8Load(ql1Ptr, vlu);
      mlir::Value qhv = u8Load(qhPtr, vlu);

      auto emitGroup = [&](mlir::Value qlByte, bool lowNibble,
                           llvm::StringRef shiftImm, int64_t aBase) {
        mlir::Value nib =
            lowNibble ? u8ImmOp("vand_vx", qlByte, "0x0F", vlu)
                      : u8ImmOp("vsrl_vx", qlByte, "0x04", vlu);
        mlir::Value qhShift =
            shiftImm == "0" ? qhv : u8ImmOp("vsrl_vx", qhv, shiftImm, vlu);
        mlir::Value qhMasked = u8ImmOp("vand_vx", qhShift, "0x03", vlu);
        mlir::Value hb = u8ImmOp("vsll_vx", qhMasked, "0x04", vlu);
        mlir::Value q6u = u8VVOp("vor_vv", nib, hb, vlu);
        std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
        mlir::Value q6i = emitOpaqueCall(rewriter, loc, cx.i8m2Type, reCallee,
                                         mlir::ValueRange{q6u}, opName, role);
        std::string subCallee = "__riscv_vsub_vx_i8m2";
        mlir::Value aVal = emitOpaqueCallBuilt(
            rewriter, loc, cx.i8m2Type, subCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value bias =
                  rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "32");
              return {q6i, bias, vlu};
            });
        mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(aChunk + aBase));
        mlir::Value dstElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, aux8Array,
                                            mlir::ValueRange{dstIdx})
                .getResult();
        mlir::Value dstPtr =
            rewriter
                .create<emitc::ApplyOp>(
                    loc, emitc::PointerType::get(cx.i8ElemType), "&", dstElem)
                .getResult();
        std::string storeCallee = "__riscv_vse8_v_i8m2";
        emitOpaqueCallVoid(rewriter, loc, storeCallee,
                           mlir::ValueRange{dstPtr, aVal, vlu}, opName, role);
      };
      // The exact _generic permutation (quants.c:828-831).
      emitGroup(ql0, /*lowNibble=*/true, "0", 0);
      emitGroup(ql1, /*lowNibble=*/true, "0x02", 32);
      emitGroup(ql0, /*lowNibble=*/false, "0x04", 64);
      emitGroup(ql1, /*lowNibble=*/false, "0x06", 96);
    }

    // ---- (B) per-sub-block int8-scaled i32 dot into the aux32 accumulator ----
    // The accumulator width follows the integer_core_lmul knob:
    //   * mf2 (default, foldGroups==1): vint32m2 (8 lanes). Each sub-block runs
    //     TWO 8-lane halves (emitHalf(0)/emitHalf(8)) summing into the same 8
    //     lanes -- the legacy byte-identical form.
    //   * m1 (foldGroups==2): vint32m4 (16 lanes). Each sub-block runs ONE
    //     16-lane strip (the whole sub-block at once); after the sub-block loop
    //     the 16 wide lanes are folded element-wise back to the canonical 8
    //     (aux32_8[l] = aux32_16[l] + aux32_16[l+8]) BEFORE the caller's fp32
    //     cvt -- VLEN-agnostically (literal element-offset vslidedown, NOT a
    //     register-subgroup vget which is VLEN128-only).
    // vint32<l32> aux32 = __riscv_vmv_v_x_i32<l32>(0, stripWidth);  (RESET/sb)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux32", opName, role));
    auto aux32Var = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(cx.i32WideType),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string aux32SeedCallee = ("__riscv_vmv_v_x_i32" + cx.l32).str();
    mlir::Value aux32Zero = emitOpaqueCallBuilt(
        rewriter, loc, cx.i32WideType, aux32SeedCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroImm =
              rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "0");
          return {zeroImm, sizeLit(cx.stripWidth)};
        });
    rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Zero);

    // The q8 / scales bases for this super-block.
    mlir::Value q8Base =
        byteOffsetPtr(yb, cx.activationPtrType, cx.q8Offset, cx.i8PtrType);
    mlir::Value scBase =
        byteOffsetPtr(xb, cx.weightPtrType, cx.scalesOffset, cx.i8PtrType);

    // for (size_t js = 0; js < 16; js += 1) { ... }
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "sub_block_loop"));
    auto subLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), sizeLit(cx.numSubBlocks), sizeLit(1),
        /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard subGuard(rewriter);
      rewriter.setInsertionPointToStart(subLoop.getBody());
      mlir::Value js = subLoop.getInductionVar();

      // int scale = (int)scales[js];  (scalar int8 sign-extended load)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scale_load"));
      mlir::Value scElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(scBase),
                  js)
              .getResult();
      mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
      mlir::Value scI8 =
          rewriter.create<emitc::LoadOp>(loc, constI8Type, scElem).getResult();
      mlir::Value scale =
          rewriter.create<emitc::CastOp>(loc, cx.i32ImmType, scI8).getResult();

      // js*16 -- the sub-block's first element offset into aux8 / q8.
      mlir::Value subBase =
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, js,
                                        sizeLit(cx.subBlock));

      // One MAC strip of `stripWidth` (8 @mf2 / 16 @m1): vwmul i8xi8 -> i16,
      // then vwmacc.vx aux32 += scale*i16. At mf2 this is the legacy 8-lane
      // half; at m1 it is the whole 16-element sub-block in one strip.
      auto emitStrip = [&](int64_t stripOffset) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_half"));
        std::string stripSetvl = ("__riscv_vsetvl_e8" + cx.l8).str();
        mlir::Value vlStrip = emitOpaqueCallBuilt(
            rewriter, loc, cx.sizeType, stripSetvl, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(cx.stripWidth)};
            });
        mlir::Value off = subBase;
        if (stripOffset != 0)
          off = rewriter.create<emitc::AddOp>(loc, cx.sizeType, subBase,
                                              sizeLit(stripOffset));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, q8Base, off)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, aux8Base, off)
                .getResult();
        std::string loadCallee = ("__riscv_vle8_v_i8" + cx.l8).str();
        mlir::Value q8v =
            emitOpaqueCall(rewriter, loc, cx.i8WideType, loadCallee,
                           mlir::ValueRange{q8Ptr, vlStrip}, opName, role);
        mlir::Value av =
            emitOpaqueCall(rewriter, loc, cx.i8WideType, loadCallee,
                           mlir::ValueRange{aPtr, vlStrip}, opName, role);
        std::string mulCallee = ("__riscv_vwmul_vv_i16" + cx.l16).str();
        mlir::Value p =
            emitOpaqueCall(rewriter, loc, cx.i16WideType, mulCallee,
                           mlir::ValueRange{q8v, av, vlStrip}, opName, role);
        std::string maccCallee = ("__riscv_vwmacc_vx_i32" + cx.l32).str();
        mlir::Value aux32Next = emitOpaqueCallBuilt(
            rewriter, loc, cx.i32WideType, maccCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aux32Cur =
                  rewriter.create<emitc::LoadOp>(loc, cx.i32WideType, aux32Var)
                      .getResult();
              return {aux32Cur, scale, p, vlStrip};
            });
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("aux32", opName, role));
        rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Next);
      };
      if (cx.foldGroups == 1) {
        // mf2: the legacy TWO 8-lane halves into the same 8 lanes (byte-exact).
        emitStrip(0);
        emitStrip(cx.half);
      } else {
        // m1: ONE 16-lane strip covering the whole 16-element sub-block.
        emitStrip(0);
      }
    }

    // ---- fold-back: collapse the wide aux32 to the canonical 8 lanes ----
    // Only emitted for foldGroups > 1 (m1). The wide accumulator holds
    // `stripWidth` lanes; group g (g=1..foldGroups-1) is element-slid down by
    // the LITERAL element offset 8*g (so element 8*g+l lands at lane l at ANY
    // VLEN -- this is what makes the fold VLEN-agnostic, unlike a vget of the
    // g-th register-subgroup whose lane count is VLEN-dependent) and added into
    // the low 8 lanes (vl=8). The canonical low 8-lane group is then vget(.,0).
    // Integer add is associative and each vwmacc stays within ONE 16-element
    // sub-block under one scalar `scale`, so aux32_8[l] = Σ_g aux32_wide[l+8g]
    // is bit-exact vs the mf2 two-half accumulation. At mf2 (foldGroups==1) this
    // block is skipped entirely -> the returned aux32 IS the 8-lane vint32m2,
    // byte-identical to today.
    mlir::TypedValue<emitc::LValueType> resultLValue =
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(aux32Var.getResult());
    if (cx.foldGroups > 1) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "aux32_fold_back"));
      mlir::Value foldWide =
          rewriter.create<emitc::LoadOp>(loc, cx.i32WideType, aux32Var)
              .getResult();
      for (int64_t g = 1; g < cx.foldGroups; ++g) {
        std::string slideCallee = ("__riscv_vslidedown_vx_i32" + cx.l32).str();
        mlir::Value slid = emitOpaqueCallBuilt(
            rewriter, loc, cx.i32WideType, slideCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value slideOff = rewriter.create<emitc::LiteralOp>(
                  loc, cx.i32ImmType, std::to_string(8 * g));
              return {foldWide, slideOff, sizeLit(8)};
            });
        std::string addCallee = ("__riscv_vadd_vv_i32" + cx.l32).str();
        foldWide = emitOpaqueCallBuilt(
            rewriter, loc, cx.i32WideType, addCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {foldWide, slid, sizeLit(8)};
            });
      }
      // vint32m2 fold = __riscv_vget_v_i32<l32>_i32m2(foldWide, 0);  -- the LOW
      // canonical 8-lane group (subgroup 0 is the low lanes at every VLEN).
      std::string getCallee =
          ("__riscv_vget_v_i32" + cx.l32 + "_i32m2").str();
      mlir::Value foldCanon = emitOpaqueCallBuilt(
          rewriter, loc, cx.i32m2Type, getCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zeroIdx =
                rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "0");
            return {foldWide, zeroIdx};
          });
      // Store into a canonical 8-lane vint32m2 lvalue the caller loads.
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("aux32_fold", opName, role));
      auto foldVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(cx.i32m2Type),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(loc, foldVar, foldCanon);
      resultLValue =
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(foldVar.getResult());
    }

    // The per-super-block aux32[8] state as the lvalue VARIABLE; the caller
    // loads it at its own statement position (K1 stores, K2 folds). At mf2 this
    // is the wide==canonical vint32m2 aux32Var; at m1 it is the folded-back
    // vint32m2 foldVar.
    return resultLValue;
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ6_KQ8_KAux32Partial(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ6KQ8KAux32Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ6KQ8KAux32Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q6_K partial body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q6_K partial ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u8Type = emitc::OpaqueType::get(ctx, "uint8_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 16
    int64_t weightStride = blockDot.getWeightBlockStride();         // 210
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qhOffset = blockDot.getWeightQhByteOffset();           // 128
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   // 192
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             // 16
    int64_t half = subBlock / 2;                      // 8

    // The integer-core vector types. The 6-bit unpack runs 32-wide chunks at
    // e8m2 (VLMAX = 32 at VLEN >= 128); the per-sub-block half-strip runs 8-wide
    // at e8m1 -> i16m1 product; the aux32 lane-collapsed accumulator is e32m2
    // (VLMAX = 8 at VLEN >= 128, exactly the 8 aux32 lanes).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination, mirroring _generic's aux8 -- the permutation lives here so the
    // per-sub-block dot reads aux8 contiguously).
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    // const int8_t *aux8 base pointer = &aux8[0] (for the vle8 contiguous reads).
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // Per-super-block base address arithmetic (vx + ib*210, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The bounded integer-core context shared with the K2 full-block-dot
    // emitter (the unpack + sub-block-loop nodes are byte-pinned by the K1
    // ssh-rvv artifact; both callers drive them identically). The K1 partial op
    // carries NO integer_core_lmul knob, so it passes the LITERAL mf2 chain
    // (l8=mf2/l16=m1/l32=m2, stripWidth=8, foldGroups=1) -- the shared helper
    // then emits the byte-identical legacy two-half form for it.
    Q6_KIntegerCoreContext cx{
        opName,        role,          sizeType,         i32ImmType,
        i8ElemType,    u8m2Type,      i8m2Type,         i8mf2Type,
        i16m1Type,     i32m2Type,     i8PtrType,        u8PtrType,
        weightPtrType, activationPtrType, subBlock,     qhOffset,
        scalesOffset,  q8Offset,      numSubBlocks,     half,
        /*coreLmul=*/"mf2", /*l8=*/"mf2", /*l16=*/"m1", /*l32=*/"m2",
        /*i8WideType=*/i8mf2Type, /*i16WideType=*/i16m1Type,
        /*i32WideType=*/i32m2Type, /*stripWidth=*/8, /*foldGroups=*/1};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K1/K2 integer core: unpack -> aux8 -> per-sub-block i32 dot,
      // returning the per-super-block aux32[8] lvalue.
      mlir::TypedValue<emitc::LValueType> aux32Var =
          emitQ6_KSuperBlockAux32Core(rewriter, loc, cx, xb, yb, aux8Array,
                                      aux8Base);

      // vse32_v_i32m2(out + ib*8, aux32, 8);  -- store the aux32[8] state.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_aux32"));
      mlir::Type i32PtrType = output.getType();
      mlir::Value outOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(8));
      mlir::Value outPtr =
          rewriter.create<emitc::AddOp>(loc, i32PtrType, output, outOff);
      mlir::Value aux32Final =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      // Void interleave (inline VL=8 literal): no full-callee void-built helper,
      // so split the mangler at the first underscore after "__riscv_" and rejoin
      // via emitVCallVoidBuilt -- byte-exact identity for "__riscv_vse32_v_i32m2".
      emitVCallVoidBuilt(rewriter, loc, "vse32", "v_i32m2", opName, role,
                         [&](mlir::OpBuilder &b, mlir::Location l)
                             -> llvm::SmallVector<mlir::Value> {
                           return {outPtr, aux32Final, sizeLit(8)};
                         });
    }

    // The op's i32 m1 result token: the lowering writes the aux32 state through
    // the output pointer (no scalar fold), so the token has no live use; bind it
    // to the zero seed literal to keep the value map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[blockDot.getResult()] = resultToken;
    (void)u8Type;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ6_KQ8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ6KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ6KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q6_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q6_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 16
    int64_t weightStride = blockDot.getWeightBlockStride();         // 210
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qhOffset = blockDot.getWeightQhByteOffset();           // 128
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   // 192
    int64_t weightDOffset = blockDot.getWeightDByteOffset();       // 208
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             // 16
    int64_t half = subBlock / 2;                      // 8
    int64_t numLanes = numSubBlocks / 2;              //   8 (aux32/sums lanes)

    // The integer-core vector types (shared with K1 via the context) + the fp32
    // fold types.
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination shared with the integer core).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // float sums8[8];  (function-scoped scratch the 8-lane fp32 accumulator
    // vector is stored into before the SEQUENTIAL horizontal sum).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 accumulator, declared + zeroed ONCE OUTSIDE the super-block loop
    // (unlike aux32, which resets each super-block; this mirrors _generic's
    // `memset(sums, 0, ...)` before the loop).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string sumsSeedCallee = "__riscv_vfmv_v_f_f32m2";
    mlir::Value sumsZero = emitOpaqueCallBuilt(
        rewriter, loc, f32m2Type, sumsSeedCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroF =
              rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
          return {zeroF, sizeLit(8)};
        });
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsZero);

    // Per-super-block base address arithmetic (vx + ib*210, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The optional integer_core_lmul knob (default "mf2" == today's byte-identical
    // emit: TWO 8-lane halves per 16-element sub-block). The Region-B MAC chain
    // widens by one notch at "m1" (ONE 16-lane strip per sub-block); "m1" is the
    // HARD CEILING (i8m1 == 16 == one sub-block; "m2" would fold two sub-blocks
    // under one scalar scale -- verifier-rejected). The 6-bit unpack (Region A)
    // and the deferred fp32 fold (Region C) are LMUL-free.
    llvm::StringRef coreLmul = blockDot.getIntegerCoreLmul().value_or("mf2");
    // l8 = the strip i8 load anchor; l16/l32 + stripWidth/foldGroups follow it up
    // the SINGLE-SOURCE widening chain (shared with the q4_K core + the codebook
    // emitters). "m1" stays the q6_K hard ceiling (the verifier rejects "m2"
    // here), so on this path the chain only ever sees {mf2,m1}.
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef l8 = wideningChain.l8;
    llvm::StringRef l16 = wideningChain.l16;
    llvm::StringRef l32 = wideningChain.l32;
    int64_t stripWidth = wideningChain.stripWidth;
    int64_t foldGroups = wideningChain.foldGroups; // 1 @mf2 (no fold), 2 @m1
    // The Region-B widened MAC types built from l8/l16/l32.
    mlir::Type i8WideType =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32WideType =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());

    // The integer-core context shared with K1 (identical unpack + sub-block-loop
    // nodes at the default mf2).
    Q6_KIntegerCoreContext cx{
        opName,        role,          sizeType,         i32ImmType,
        i8ElemType,    u8m2Type,      i8m2Type,         i8mf2Type,
        i16m1Type,     i32m2Type,     i8PtrType,        u8PtrType,
        weightPtrType, activationPtrType, subBlock,     qhOffset,
        scalesOffset,  q8Offset,      numSubBlocks,     half,
        coreLmul,      l8,            l16,              l32,
        i8WideType,    i16WideType,   i32WideType,      stripWidth,
        foldGroups};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K1/K2 integer core: unpack -> aux8 -> per-sub-block i32 dot,
      // returning the per-super-block aux32[8] lvalue.
      mlir::TypedValue<emitc::LValueType> aux32Var =
          emitQ6_KSuperBlockAux32Core(rewriter, loc, cx, xb, yb, aux8Array,
                                      aux8Base);

      // ---- (C) the DEFERRED two-level fp32 fold ----
      // float d = (float)*(const _Float16 *)(xb + 208) * *(const float *)(yb+0);
      // The weight super-block scale is fp16 (read via the one sanctioned opaque
      // seam); the activation scale is fp32 (a plain float load).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      // *(const float *)(yb + 0): cast the activation base to const float * and
      // load element 0 (the fp32 q8_K scale d).
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();
      // d = dx * dy  (the fp16->fp32 weight scale times the fp32 activation
      // scale, a single fp32 multiply matching _generic's `const float d = ...`).
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
      //   ((float)aux32[l], RNE -- matches C's `(float)aux32[l]` lane-wise)
      mlir::Value aux32Val =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
      mlir::Value af = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, cvtCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {aux32Val, sizeLit(8)};
          });
      // vfloat32m2_t pr = __riscv_vfmul_vf_f32m2(af, d, 8);  -- SEPARATE multiply
      // (NEVER a fused vfmacc: byte-exact only if the mul and add stay distinct,
      // matching _generic's scalar `d * aux32[l]` then `sums[l] += ...`).
      std::string mulCallee = "__riscv_vfmul_vf_f32m2";
      mlir::Value pr = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, mulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {af, d, sizeLit(8)};
          });
      // sums = __riscv_vfadd_vv_f32m2(sums, pr, 8);  -- SEPARATE add (NEVER fma).
      std::string addCallee = "__riscv_vfadd_vv_f32m2";
      mlir::Value sumsNext = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, addCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumsCur =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar)
                    .getResult();
            return {sumsCur, pr, sizeLit(8)};
          });
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sums", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);
    }

    // ---- (D) the SEQUENTIAL horizontal sum, l = 0 .. 7 ----
    // vse32_v_f32m2(&sums8[0], sums, 8);  -- materialize lane l at sums8[l]
    // (the lane <-> index correspondence the sequential sum relies on).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums_lanes"));
    mlir::Value sums8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sums8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sums8Array,
                                        mlir::ValueRange{sums8Index0})
            .getResult();
    mlir::Value sums8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(floatType), "&",
                                    sums8Elem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    emitVCallVoidBuilt(rewriter, loc, "vse32", "v_f32m2", opName, role,
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         return {sums8Base, sumsFinal, sizeLit(8)};
                       });

    // float sumf = 0.0f; sumf += sums8[0]; sumf += sums8[1]; ...; sumf +=
    // sums8[7];  -- the SEQUENTIAL ascending fp32 horizontal sum (NEVER a
    // vfredusum: fp add is non-associative; this exactly mirrors _generic's
    // `float sumf = 0; for (l=0..7) sumf += sums[l]` INCLUDING the leading
    // `0.0f +` -- 8 adds from l=0, so byte-exactness holds even at the -0.0
    // edge).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto loadLane = [&](int64_t l) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(l));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(loc, sums8Array, mlir::ValueRange{idx})
              .getResult();
      return rewriter.create<emitc::LoadOp>(loc, floatType, elem).getResult();
    };
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "horizontal_sum"));
    mlir::Value sumf =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f").getResult();
    for (int64_t l = 0; l < numLanes; ++l)
      sumf =
          rewriter.create<emitc::AddOp>(loc, floatType, sumf, loadLane(l))
              .getResult();

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q6_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

VariantToEmitCFunc::Q4_KCoreResult VariantToEmitCFunc::emitQ4_KSuperBlockAux32Core(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base,
    mlir::TypedValue<emitc::ArrayType> utmpArray, mlir::Value scaleMinOutput,
    mlir::Value ib) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    int64_t qk = cx.subBlock * cx.numSubBlocks; // 256
    // NB: the legacy `quarter` local (== cx.quarter == 8) is gone — the strip
    // offset is now derived from cx.stripWidth (mf2 -> 8 == the old quarter).

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    auto u32Lit = [&](llvm::StringRef v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.u32Type, v.str());
    };
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };

    // ---- (A) 4-bit nibble unpack into aux8 (element-ordered, NO bias) ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "unpack_4bit"));
    std::string unpackSetvl = "__riscv_vsetvl_e8m2";
    auto u8Load = [&](mlir::Value ptr, mlir::Value vl) -> mlir::Value {
      return emitVCall(rewriter, loc, cx.u8m2Type, "vle8_v", "u8m2",
                       mlir::ValueRange{ptr, vl}, opName, role);
    };
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef imm, mlir::Value vl) -> mlir::Value {
      return emitVCallBuilt(
          rewriter, loc, cx.u8m2Type, mnemonic, "u8m2", opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, cx.i32ImmType, imm.str());
            return {src, amt, vl};
          });
    };
    auto u8AddVV = [&](mlir::Value a, mlir::Value b,
                       mlir::Value vl) -> mlir::Value {
      return emitVCall(rewriter, loc, cx.u8m2Type, "vadd_vv", "u8m2",
                       mlir::ValueRange{a, b, vl}, opName, role);
    };
    for (int64_t chunk = 0; chunk < qk / 64; ++chunk) {
      int64_t qsChunk = chunk * 32; // q4 advances 32 bytes per 64-elem chunk
      int64_t aChunk = chunk * 64;  // aux8 element base for this chunk
      mlir::Value vlu = emitOpaqueCallBuilt(
          rewriter, loc, cx.sizeType, unpackSetvl, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(32)};
          });
      mlir::Value qsPtr =
          byteOffsetPtr(xb, cx.weightPtrType, cx.qsOffset + qsChunk,
                        cx.u8PtrType);
      mlir::Value q4 = u8Load(qsPtr, vlu);

      // (q5_K only) the qh high-bit plane (32 bytes @ qhOffset) loaded ONCE per
      // super-block from a FIXED pointer (NOT chunk-strided -- `hm = x[i].qh` is
      // reused across all 8 halves in _generic; only the tested bit `m` varies).
      // Loaded inside the chunk loop because the e8m2 32-lane VLMAX matches the
      // 32-element qh span; q5_K's qh[l] (l=0..31) is identical for both halves
      // of every chunk, so reloading the same 32 bytes each chunk is exact.
      mlir::Value qh;
      if (cx.hasQh) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_high_bit_plane"));
        mlir::Value qhPtr =
            byteOffsetPtr(xb, cx.weightPtrType, cx.qhOffset, cx.u8PtrType);
        qh = u8Load(qhPtr, vlu);
      }

      // The qh 5th-bit injection for one 32-element half h in 0..7: add
      // `((qh[l] >> h) & 1) << 4` (== 16 iff qh bit h of element l is set) to the
      // unpacked nibble in the UINT8 domain, exactly mirroring _generic's
      // `a[l] += (hm[l] & m ? 16 : 0)` with m = 1<<h. Done BEFORE the u8->i8
      // reinterpret (q5 in [0,31] stays non-negative so the reinterpret is exact).
      auto injectQh = [&](mlir::Value nib, int64_t h) -> mlir::Value {
        mlir::Value shifted =
            (h == 0) ? qh : u8ImmOp("vsrl_vx", qh, std::to_string(h), vlu);
        mlir::Value bit = u8ImmOp("vand_vx", shifted, "0x01", vlu);
        mlir::Value contrib = u8ImmOp("vsll_vx", bit, "0x04", vlu);
        return u8AddVV(nib, contrib, vlu);
      };

      auto emitNibble = [&](bool lowNibble, int64_t aBase) {
        mlir::Value nib =
            lowNibble ? u8ImmOp("vand_vx", q4, "0x0F", vlu)
                      : u8ImmOp("vsrl_vx", q4, "0x04", vlu);
        if (cx.hasQh) {
          // Half index h = chunk*2 + (lowNibble ? 0 : 1): _generic advances m by
          // <<1 per 32-element half (low then high), so half h tests bit h.
          int64_t h = chunk * 2 + (lowNibble ? 0 : 1);
          nib = injectQh(nib, h);
        }
        std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
        mlir::Value q4i = emitOpaqueCall(rewriter, loc, cx.i8m2Type, reCallee,
                                         mlir::ValueRange{nib}, opName, role);
        mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(aChunk + aBase));
        mlir::Value dstElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, aux8Array,
                                            mlir::ValueRange{dstIdx})
                .getResult();
        mlir::Value dstPtr =
            rewriter
                .create<emitc::ApplyOp>(
                    loc, emitc::PointerType::get(cx.i8ElemType), "&", dstElem)
                .getResult();
        std::string storeCallee = "__riscv_vse8_v_i8m2";
        emitOpaqueCallVoid(rewriter, loc, storeCallee,
                           mlir::ValueRange{dstPtr, q4i, vlu}, opName, role);
      };
      emitNibble(/*lowNibble=*/true, 0);   // a[0..31]  = q4[l] & 0xF
      emitNibble(/*lowNibble=*/false, 32); // a[32..63] = q4[l] >> 4
    }

    // ---- (B) the 6-bit scale/min bit-dance (utmp/kmask), STRUCTURED scalar
    // emitc.bitwise ops (NO raw string), mirroring _generic (quants.c:685-690).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "scale_min_bit_dance"));
    mlir::Value kmask1 = u32Lit("0x3f3f3f3f");
    mlir::Value kmask2 = u32Lit("0x0f0f0f0f");
    mlir::Value kmask3 = u32Lit("0x03030303");
    mlir::Value shift4 = u32Lit("4");
    mlir::Value shift6 = u32Lit("6");
    mlir::Value scWordPtr =
        byteOffsetPtr(xb, cx.weightPtrType, cx.scalesOffset, cx.u32PtrType);
    auto loadWord = [&](int64_t idx) -> mlir::Value {
      mlir::Value wIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(idx));
      mlir::Value wElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(scWordPtr),
                  wIdx)
              .getResult();
      mlir::Value cw =
          rewriter.create<emitc::LoadOp>(loc, cx.constU32Type, wElem)
              .getResult();
      return rewriter.create<emitc::CastOp>(loc, cx.u32Type, cw).getResult();
    };
    mlir::Value w0 = loadWord(0);
    mlir::Value w1 = loadWord(1);
    mlir::Value w2 = loadWord(2);
    auto bAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseAndOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    auto bOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseOrOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    auto bShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseRightShiftOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    auto bShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    mlir::Value u3 =
        bOr(bAnd(bShr(w2, shift4), kmask2),
            bShl(bAnd(bShr(w1, shift6), kmask3), shift4));
    mlir::Value u2 = bAnd(w1, kmask1);
    mlir::Value u1 =
        bOr(bAnd(w2, kmask2),
            bShl(bAnd(bShr(w0, shift6), kmask3), shift4));
    mlir::Value u0 = bAnd(w0, kmask1);
    auto storeUtmp = [&](int64_t idx, mlir::Value v) {
      mlir::Value uIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(idx));
      mlir::Value uElem =
          rewriter
              .create<emitc::SubscriptOp>(loc, utmpArray,
                                          mlir::ValueRange{uIdx})
              .getResult();
      rewriter.create<emitc::AssignOp>(
          loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(uElem), v);
    };
    storeUtmp(0, u0);
    storeUtmp(1, u1);
    storeUtmp(2, u2);
    storeUtmp(3, u3);

    // scales = (const uint8_t *)&utmp[0]  (the 8 6-bit scales, contiguous with
    // the 8 mins -- 16 bytes total).
    mlir::Value utmpIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value utmpElem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, utmpArray,
                                        mlir::ValueRange{utmpIndex0})
            .getResult();
    mlir::Value utmpWordPtr =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(cx.u32Type),
                                    "&", utmpElem0)
            .getResult();
    mlir::Value scalesU8 =
        rewriter.create<emitc::CastOp>(loc, cx.u8PtrType, utmpWordPtr)
            .getResult();

    // (K4a only) vse8 the 16 decoded [scales[0..7], mins[0..7]] bytes to
    // scaleMinOutput + ib*16 -- emitted IN-PLACE here (before the aux32 init) so
    // K4a's node sequence is unchanged. K4b passes a null scaleMinOutput.
    if (scaleMinOutput) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_scale_min"));
      std::string smSetvl = "__riscv_vsetvl_e8m1";
      mlir::Value vlsm = emitOpaqueCallBuilt(
          rewriter, loc, cx.sizeType, smSetvl, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(16)};
          });
      std::string smLoad = "__riscv_vle8_v_u8m1";
      mlir::Type u8m1Type = emitc::OpaqueType::get(ctx, "vuint8m1_t");
      mlir::Value smVec =
          emitOpaqueCall(rewriter, loc, u8m1Type, smLoad,
                         mlir::ValueRange{scalesU8, vlsm}, opName, role);
      mlir::Type smOutPtrType = scaleMinOutput.getType();
      mlir::Value smOff =
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, ib, sizeLit(16));
      mlir::Value smOutPtr = rewriter.create<emitc::AddOp>(
          loc, smOutPtrType, scaleMinOutput, smOff);
      std::string smStore = "__riscv_vse8_v_u8m1";
      emitOpaqueCallVoid(rewriter, loc, smStore,
                         mlir::ValueRange{smOutPtr, smVec, vlsm}, opName, role);
    }

    // ---- (C) per-sub-block uint6-scaled i32 dot into the 8-lane aux32 ----
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux32", opName, role));
    auto aux32Var = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(cx.i32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    // Region-C MAC chain callees keyed off the integer_core_lmul anchor
    // (cx.l8/l16/l32). At the default mf2 -> m1 -> m2 these render the exact
    // legacy callee strings (vmv_v_x_i32m2 / vsetvl_e8mf2 / vle8_v_i8mf2 /
    // vwmul_vv_i16m1 / vwmacc_vx_i32m2) with vl == stripWidth == 8. At m1/m2 the
    // running aux32 is the WIDE vint32<l32> over stripWidth lanes (16/32); the
    // fold-back (Region-C tail) collapses it to the canonical 8 before Region F.
    std::string aux32SeedCallee = ("__riscv_vmv_v_x_i32" + cx.l32).str();
    mlir::Value aux32Zero = emitOpaqueCallBuilt(
        rewriter, loc, cx.i32m2Type, aux32SeedCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroImm =
              rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "0");
          return {zeroImm, sizeLit(cx.stripWidth)};
        });
    rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Zero);

    mlir::Value q8Base =
        byteOffsetPtr(yb, cx.activationPtrType, cx.q8Offset, cx.i8PtrType);

    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "sub_block_loop"));
    auto subLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), sizeLit(cx.numSubBlocks), sizeLit(1),
        /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard subGuard(rewriter);
      rewriter.setInsertionPointToStart(subLoop.getBody());
      mlir::Value js = subLoop.getInductionVar();

      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scale_load"));
      mlir::Value scElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(scalesU8),
                  js)
              .getResult();
      mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
      mlir::Value scU8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, scElem).getResult();
      mlir::Value scale =
          rewriter.create<emitc::CastOp>(loc, cx.i32ImmType, scU8).getResult();

      mlir::Value subBase =
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, js,
                                        sizeLit(cx.subBlock));

      // One MAC strip of cx.stripWidth elements at byte offset stripOffset into
      // the sub-block: vsetvl_e8<l8>(stripWidth) -> two vle8 (q8 + aux8) ->
      // vwmul i16<l16> -> vwmacc i32<l32> into the running (wide) aux32. At mf2
      // stripWidth==8 == the legacy quarter, 4 strips/sub-block; at m1/m2 the
      // strip is 16/32-wide (2/1 strips), writing the wide aux32's upper lanes.
      auto emitStrip = [&](int64_t stripOffset) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_quarter"));
        std::string quarterSetvl = ("__riscv_vsetvl_e8" + cx.l8).str();
        mlir::Value vl8 = emitOpaqueCallBuilt(
            rewriter, loc, cx.sizeType, quarterSetvl, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(cx.stripWidth)};
            });
        mlir::Value off = subBase;
        if (stripOffset != 0)
          off = rewriter.create<emitc::AddOp>(loc, cx.sizeType, subBase,
                                              sizeLit(stripOffset));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, q8Base, off)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, aux8Base, off)
                .getResult();
        std::string loadCallee = ("__riscv_vle8_v_i8" + cx.l8).str();
        mlir::Value q8v =
            emitOpaqueCall(rewriter, loc, cx.i8mf2Type, loadCallee,
                           mlir::ValueRange{q8Ptr, vl8}, opName, role);
        mlir::Value av =
            emitOpaqueCall(rewriter, loc, cx.i8mf2Type, loadCallee,
                           mlir::ValueRange{aPtr, vl8}, opName, role);
        std::string mulCallee = ("__riscv_vwmul_vv_i16" + cx.l16).str();
        mlir::Value p =
            emitOpaqueCall(rewriter, loc, cx.i16m1Type, mulCallee,
                           mlir::ValueRange{q8v, av, vl8}, opName, role);
        std::string maccCallee = ("__riscv_vwmacc_vx_i32" + cx.l32).str();
        mlir::Value aux32Next = emitOpaqueCallBuilt(
            rewriter, loc, cx.i32m2Type, maccCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value aux32Cur =
                  rewriter.create<emitc::LoadOp>(loc, cx.i32m2Type, aux32Var)
                      .getResult();
              return {aux32Cur, scale, p, vl8};
            });
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("aux32", opName, role));
        rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Next);
      };
      for (int64_t s = 0; s < cx.numStrips; ++s)
        emitStrip(s * cx.stripWidth);
    }

    // ---- (C-tail) integer fold-back of the WIDE aux32 to the canonical 8 ----
    // At mf2 (foldGroups == 1) the running aux32 IS already the canonical 8-lane
    // vint32m2; nothing is emitted and the result is byte-identical to today. At
    // m1/m2 the wide aux32 holds stripWidth = 16/32 lanes whose group-of-8
    // residues must be summed: aux32_8[l] = Sum_k aux32_wide[l + 8*k], k in
    // [0,foldGroups). Integer add is associative/order-free, and within each
    // strip the per-sub-block `scale` was folded in during the vwmacc (constant
    // across the lanes being summed) -- so this regroup is provably bit-exact at
    // every legal LMUL (DESIGN sec 0). Emitted REGISTER-ONLY via vget (extract
    // the k-th 8-lane m2 subgroup from the wide register group) + vadd -- NO
    // memory spill (the per-super-block round-trip a vse/vle pair would cost),
    // keeping the canonical 8-lane vint32m2 that Region F consumes.
    mlir::TypedValue<emitc::LValueType> resultAux32Var =
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(aux32Var.getResult());
    if (cx.foldGroups > 1) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "aux32_fold_back_to_8"));
      mlir::Value aux32WideVal =
          rewriter.create<emitc::LoadOp>(loc, cx.i32m2Type, aux32Var).getResult();
      // VLEN-AGNOSTIC regroup (the prior vget(.,k) form was VLEN128-ONLY): the
      // k-th group occupies ELEMENT window [8*k, 8*k+8) of the wide register --
      // NOT the k-th vget register-subgroup (whose lane count = LMUL*VLEN/SEW =
      // VLEN/16 = 8 at VLEN128 but 16 at VLEN256, so vget(.,k) reads [16k,16k+16)
      // and folds the WRONG lanes at VLEN256). We instead vslidedown the wide
      // register by the LITERAL element offset 8*k (k = 1..foldGroups-1) so
      // element 8*k+l lands at lane l, then vadd at vl==8 IN THE WIDE TYPE --
      // both ops are element-indexed and so VLEN-correct at any VLEN. The final
      // vget(.,0) extracts the low canonical 8-lane group (lanes [0,8), always
      // correct -- the byte-exact Region-F type). Slide offsets max at 8*3=24 at
      // m2 (reads [24,32) ⊂ [0,32), all written) -- never touches seed tail.
      std::string slideCallee = ("__riscv_vslidedown_vx_i32" + cx.l32).str();
      std::string wideAddCallee = ("__riscv_vadd_vv_i32" + cx.l32).str();
      std::string getCallee = ("__riscv_vget_v_i32" + cx.l32 + "_i32m2").str();
      mlir::Value foldWide = aux32WideVal; // k=0 group = lanes [0,8)
      for (int64_t k = 1; k < cx.foldGroups; ++k) {
        mlir::Value slid = emitOpaqueCallBuilt(
            rewriter, loc, cx.i32m2Type, slideCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {aux32WideVal, sizeLit(8 * k), sizeLit(8)};
            });
        foldWide = emitOpaqueCallBuilt(
            rewriter, loc, cx.i32m2Type, wideAddCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {foldWide, slid, sizeLit(8)};
            });
      }
      // vint32m2_t fold = __riscv_vget_v_i32<l32>_i32m2(foldWide, 0);  -- the low
      // canonical 8-lane group (lanes [0,8)); the accumulated residues now live
      // there. Subgroup 0 is the low lanes at every VLEN, so this extract is safe.
      mlir::Value fold = emitOpaqueCallBuilt(
          rewriter, loc, cx.i32Canon8Type, getCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zeroImm =
                rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "0");
            return {foldWide, zeroImm};
          });
      // vint32m2_t aux32c = <fold>;  -- the canonical 8-lane accumulator the fp
      // fold (Region F) consumes, replacing the wide aux32 lvalue.
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("aux32c", opName, role));
      auto aux32cVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(cx.i32Canon8Type),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(loc, aux32cVar, fold);
      resultAux32Var =
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(aux32cVar.getResult());
    }

    return Q4_KCoreResult{resultAux32Var, scalesU8};
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KQ8_KAux32Partial(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ4KQ8KAux32Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ4KQ8KAux32Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K partial body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value aux32Output = valueMap.lookup(blockDot.getAux32Output());
    mlir::Value scaleMinOutput = valueMap.lookup(blockDot.getScaleminOutput());
    if (!weightBase || !activationBase || !aux32Output || !scaleMinOutput)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q4_K partial ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 32
    int64_t weightStride = blockDot.getWeightBlockStride();         // 144
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   //   4
    int64_t qsOffset = blockDot.getWeightQsByteOffset();           //  16
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             //   8
    int64_t quarter = 8;                              // 8-elem quarters

    // The integer-core vector types. The 4-bit unpack runs 32-wide chunks at
    // e8m2 (VLMAX = 32 at VLEN >= 128); the per-sub-block quarter-strip runs
    // 8-wide at e8mf2 -> i16m1 product; the aux32 lane-collapsed accumulator is
    // e32m2 (VLMAX = 8 at VLEN >= 128, exactly the 8 aux32 lanes).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    mlir::Type u32PtrType =
        emitc::PointerType::get(constU32Type);

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    // The integer-core context shared with K4b (identical unpack + bit-dance +
    // per-sub-block-loop nodes).
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        activationPtrType, subBlock,  scalesOffset,  qsOffset,
        q8Offset,      numSubBlocks,  quarter};

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination, mirroring _generic's aux8 -- the layout lives here so the
    // per-sub-block dot reads aux8 contiguously).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // uint32_t utmp[4];  (function-scoped scratch for the bit-dance; the 16 bytes
    // utmp[0..3] laid out are exactly [scales[0..7], mins[0..7]] -- the same
    // type-pun _generic does with scales=&utmp[0]/mins=&utmp[2]).
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // Per-super-block base address arithmetic (vx + ib*144, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K4a/K4b integer core (the 4-bit unpack + the bit-dance + the
      // in-place 16-byte scale/min store gated on the non-null scaleMinOutput +
      // the per-sub-block i32 dot), returning the per-super-block aux32 lvalue.
      Q4_KCoreResult core = emitQ4_KSuperBlockAux32Core(
          rewriter, loc, cx, xb, yb, aux8Array, aux8Base, utmpArray,
          scaleMinOutput, ib);
      mlir::TypedValue<emitc::LValueType> aux32Var = core.aux32Var;

      // vse32_v_i32m2(aux32_out + ib*8, aux32, 8);  -- store the aux32[8] state.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_aux32"));
      mlir::Type i32PtrType = aux32Output.getType();
      mlir::Value outOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(8));
      mlir::Value outPtr =
          rewriter.create<emitc::AddOp>(loc, i32PtrType, aux32Output, outOff);
      mlir::Value aux32Final =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      // Void interleave (inline VL=8 literal): split the mangler at the first
      // underscore after "__riscv_" and rejoin via emitVCallVoidBuilt -- byte-
      // exact identity for "__riscv_vse32_v_i32m2".
      emitVCallVoidBuilt(rewriter, loc, "vse32", "v_i32m2", opName, role,
                         [&](mlir::OpBuilder &b, mlir::Location l)
                             -> llvm::SmallVector<mlir::Value> {
                           return {outPtr, aux32Final, sizeLit(8)};
                         });
    }

    // The op's i32 m1 result token: the lowering writes the aux32/scalemin state
    // through the output pointers (no scalar fold), so the token has no live use;
    // bind it to the zero seed literal to keep the value map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[blockDot.getResult()] = resultToken;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KQ8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ4KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ4KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q4_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 32
    int64_t weightStride = blockDot.getWeightBlockStride();         // 144
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();       //   0
    int64_t weightDminOffset = blockDot.getWeightDminByteOffset(); //   2
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   //   4
    int64_t qsOffset = blockDot.getWeightQsByteOffset();           //  16
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t bsumsOffset = blockDot.getActivationBsumsByteOffset(); // 260
    int64_t numSubBlocks = qk / subBlock;             //   8
    int64_t quarter = 8;                              // 8-elem quarters
    int64_t numLanes = 8;                             // aux32/sums lanes
    int64_t numBsums = qk / 16;                       //  16

    // The Region-C integer-MAC LMUL anchor, sourced from the optional
    // integer_core_lmul (default "mf2" == today's byte-identical emit). l8/l16/l32
    // are the three rungs of the i8 -> i16 -> i32 widening chain; the default
    // (mf2 -> m1 -> m2) reproduces the legacy callee/type strings exactly. The
    // Region-A 6-bit unpack (u8m2/i8m2) is NOT on this knob -- it stays fixed.
    llvm::StringRef coreLmul = blockDot.getIntegerCoreLmul().value_or("mf2");
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef l8 = wideningChain.l8;                                // base
    llvm::StringRef l16 = wideningChain.l16;
    llvm::StringRef l32 = wideningChain.l32;
    // S3-S4: the Region-C MAC strip width follows l8 (the i8 base LMUL): mf2 is
    // the 8-wide quarter strip (today, 4 strips/sub-block); m1 a 16-wide strip
    // (2/sub-block); m2 a 32-wide strip (1/sub-block == the whole sub-block in
    // one MAC). The wider strip writes the upper aux32 lanes, then the integer
    // fold-back (Region C tail) collapses those foldGroups = stripWidth/8 lane
    // groups back to the canonical 8 BEFORE the byte-exact fp fold (Region F),
    // which stays 8-lane vint32m2_t forever.
    int64_t stripWidth = wideningChain.stripWidth;
    int64_t numStrips = subBlock / stripWidth; // 32/8=4, 32/16=2, 32/32=1
    int64_t foldGroups = wideningChain.foldGroups; // 1 (mf2), 2 (m1), 4 (m2)

    // The integer-core vector types (shared with K4a via the context) + the fp32
    // fold types. The Region-A unpack types (u8m2/i8m2) are fixed at m2; the
    // Region-C MAC types are derived from the l8/l16/l32 chain.
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    // The CANONICAL 8-lane integer accumulator type the wide aux32 folds back
    // into and the fp fold (Region F) consumes -- ALWAYS vint32m2_t (8 lanes at
    // SEW32, the byte-exact contract). At mf2 it equals i32m2Type (no widen).
    mlir::Type i32Canon8Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    mlir::Type u32PtrType = emitc::PointerType::get(constU32Type);
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constI16Type = emitc::OpaqueType::get(ctx, "const int16_t");
    mlir::Type constI16PtrType = emitc::PointerType::get(constI16Type);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination shared with the integer core).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // uint32_t utmp[4];  (function-scoped scratch for the bit-dance; the 16 bytes
    // utmp[0..3] are exactly [scales[0..7], mins[0..7]]).
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // float sums8[8];  (function-scoped scratch the 8-lane fp32 accumulator
    // vector is stored into before the SEQUENTIAL horizontal sum).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 positive accumulator, declared + zeroed ONCE OUTSIDE the super-block
    // loop (unlike aux32, which resets each super-block; mirrors _generic's
    // `memset(sums, 0, ...)`).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string sumsSeedCallee = "__riscv_vfmv_v_f_f32m2";
    mlir::Value sumsZero = emitOpaqueCallBuilt(
        rewriter, loc, f32m2Type, sumsSeedCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroF =
              rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
          return {zeroF, sizeLit(8)};
        });
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsZero);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator the MIN
    // subtractions land in IN-LOOP (in super-block order); declared + zeroed ONCE
    // OUTSIDE the loop (distinct from q6_K's post-loop-only sumf -- mirrors
    // _generic's `float sumf = 0;` before the super-block loop).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // Per-super-block base address arithmetic (vx + ib*144, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The integer-core context shared with K4a (identical unpack + bit-dance +
    // per-sub-block-loop nodes). K4b passes a null scaleMinOutput / ib so the
    // helper does NOT emit the K4a-only 16-byte scale/min store. The Region-C
    // LMUL chain (coreLmul/l8/l16/l32) flows through the context to the helper.
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        activationPtrType, subBlock,  scalesOffset,  qsOffset,
        q8Offset,      numSubBlocks,  quarter,       coreLmul,
        l8,            l16,           l32,           stripWidth,
        numStrips,     foldGroups,    i32Canon8Type};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K4a/K4b integer core: unpack + bit-dance + per-sub-block i32
      // dot, returning the per-super-block aux32[8] lvalue AND the decoded
      // scales/mins pointer (mins = scalesU8 + 8). K4b passes a null
      // scaleMinOutput so the helper elides the K4a 16-byte scale/min store.
      Q4_KCoreResult core = emitQ4_KSuperBlockAux32Core(
          rewriter, loc, cx, xb, yb, aux8Array, aux8Base, utmpArray,
          /*scaleMinOutput=*/nullptr, /*ib=*/nullptr);
      mlir::TypedValue<emitc::LValueType> aux32Var = core.aux32Var;
      mlir::Value scalesU8 = core.scalesU8;

      // The fp32 activation scale dy = *(const float *)(yb + 0), loaded ONCE and
      // reused by both the positive d and the min-term dmin (mirrors _generic
      // reading `y[i].d` once).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_activation_d"));
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();

      // ---- the q4_K MIN term (the new piece vs q6_K) ----
      // int sumi = 0; for (j=0..15) sumi += (int)bsums[j] * (int)mins[j/2];
      // bsums are int16 (SIGN-extended on load); mins are the decoded uint6
      // scalesU8[8..15] (mins = scalesU8 + 8); each min spans TWO consecutive
      // bsums (mins[j/2]). Integer multiply/add -> order-free, a scalar loop.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "min_term_bsums"));
      mlir::Value bsumsAddr = yb;
      if (bsumsOffset != 0)
        bsumsAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                  sizeLit(bsumsOffset));
      mlir::Value bsumsPtr =
          rewriter.create<emitc::CastOp>(loc, constI16PtrType, bsumsAddr)
              .getResult();
      // int sumi = 0;  (the per-super-block scalar min accumulator).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value sumiZero =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
      rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiZero);
      for (int64_t j = 0; j < numBsums; ++j) {
        // (int)bsums[j]  -- int16 sign-extended to int.
        mlir::Value bIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(j));
        mlir::Value bElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(bsumsPtr),
                    bIdx)
                .getResult();
        mlir::Value bs16 =
            rewriter.create<emitc::LoadOp>(loc, constI16Type, bElem)
                .getResult();
        mlir::Value bs =
            rewriter.create<emitc::CastOp>(loc, i32Type, bs16).getResult();
        // (int)mins[j/2]  -- the decoded uint6 min, scalesU8[8 + j/2].
        mlir::Value mIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(8 + j / 2));
        mlir::Value mElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(scalesU8),
                    mIdx)
                .getResult();
        mlir::Value mU8 =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, mElem).getResult();
        mlir::Value m =
            rewriter.create<emitc::CastOp>(loc, i32Type, mU8).getResult();
        // sumi += bsums[j] * mins[j/2];  (integer, order-free).
        mlir::Value prod =
            rewriter.create<emitc::MulOp>(loc, i32Type, bs, m).getResult();
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
        mlir::Value sumiNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumiCur, prod)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
      }

      // ---- the DEFERRED two-level fp32 positive fold (q6_K K2 mechanism) ----
      // float d = (float)*(const _Float16 *)(xb + 0) * dy;  -- the fp16 weight
      // super-block scale (byte 0) times the fp32 activation scale (loaded once).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
      // The helper returns the CANONICAL 8-lane vint32m2 (the S4 fold-back
      // collapses the wide aux32 at m1/m2; at mf2/q5_K it is the running aux32,
      // same type) -- load it as i32Canon8Type (always vint32m2_t), NOT the wide
      // i32m2Type (vint32<l32>), byte-identical at mf2 and type-correct at m1/m2.
      mlir::Value aux32Val =
          rewriter.create<emitc::LoadOp>(loc, i32Canon8Type, aux32Var)
              .getResult();
      std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
      mlir::Value af = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, cvtCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {aux32Val, sizeLit(8)};
          });
      // vfloat32m2_t pr = __riscv_vfmul_vf_f32m2(af, d, 8);  -- SEPARATE multiply.
      std::string mulCallee = "__riscv_vfmul_vf_f32m2";
      mlir::Value pr = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, mulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {af, d, sizeLit(8)};
          });
      // sums = __riscv_vfadd_vv_f32m2(sums, pr, 8);  -- SEPARATE add (NEVER fma).
      std::string addCallee = "__riscv_vfadd_vv_f32m2";
      mlir::Value sumsNext = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, addCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumsCur =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar)
                    .getResult();
            return {sumsCur, pr, sizeLit(8)};
          });
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sums", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);

      // ---- the MIN subtraction: sumf -= dmin * (float)sumi ----
      // float dmin = (float)*(const _Float16 *)(xb + 2) * dy;  -- the fp16 weight
      // min scale (byte 2) times the SAME fp32 activation scale.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_dmin"));
      mlir::Value dmAddr = xb;
      if (weightDminOffset != 0)
        dmAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDminOffset));
      mlir::Value dmx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                       mlir::ValueRange{dmAddr}, opName, role,
                                       llvm::StringRef("fcvt.s.h"));
      mlir::Value dmin =
          rewriter.create<emitc::MulOp>(loc, floatType, dmx, dy).getResult();
      // sumf = sumf - dmin * (float)sumi;  -- ONE emitc.expression so it renders
      // as ggml's single C statement (quants.c:714 `sumf -= dmin * sumi`) and
      // tracks its contraction. The emitc.load temps stay OUTSIDE the expression.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "min_subtract"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto minExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&minExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // dmin * (float)sumi  -- the min product (ggml binds `dmin * sumi`
        // before the `-=`).
        mlir::Value minProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dmin, sumiFloat);
        // sumf - (dmin * sumi)  -- the `-=` subtraction tree.
        mlir::Value sumfNext =
            rewriter.create<emitc::SubOp>(loc, floatType, sumfCur, minProduct);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, minExpr.getResult());
    }

    // ---- the SEQUENTIAL horizontal sum, l = 0 .. 7 (AFTER the super-block loop;
    // sumf already holds the accumulated MIN subtractions) ----
    // vse32_v_f32m2(&sums8[0], sums, 8);  -- materialize lane l at sums8[l].
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums_lanes"));
    mlir::Value sums8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sums8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sums8Array,
                                        mlir::ValueRange{sums8Index0})
            .getResult();
    mlir::Value sums8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(floatType), "&",
                                    sums8Elem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    emitVCallVoidBuilt(rewriter, loc, "vse32", "v_f32m2", opName, role,
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         return {sums8Base, sumsFinal, sizeLit(8)};
                       });

    // sumf += sums8[0]; sumf += sums8[1]; ...; sumf += sums8[7];  -- the
    // SEQUENTIAL ascending fp32 horizontal sum, STARTING from the carried sumf
    // (which holds the MIN subtractions), NEVER a vfredusum. This mirrors
    // _generic's `for (l=0..7) sumf += sums[l]` after the super-block loop, with
    // sumf already carrying the in-loop min subtractions.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "horizontal_sum"));
    auto loadLane = [&](int64_t l) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(l));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(loc, sums8Array, mlir::ValueRange{idx})
              .getResult();
      return rewriter.create<emitc::LoadOp>(loc, floatType, elem).getResult();
    };
    mlir::Value sumf =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    for (int64_t l = 0; l < numLanes; ++l)
      sumf =
          rewriter.create<emitc::AddOp>(loc, floatType, sumf, loadLane(l))
              .getResult();

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q4_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ5_KQ8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ5KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ5KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q5_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q5_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 32
    int64_t weightStride = blockDot.getWeightBlockStride();         // 176
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();       //   0
    int64_t weightDminOffset = blockDot.getWeightDminByteOffset(); //   2
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   //   4
    int64_t qhOffset = blockDot.getWeightQhByteOffset();           //  16
    int64_t qsOffset = blockDot.getWeightQsByteOffset();           //  48
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t bsumsOffset = blockDot.getActivationBsumsByteOffset(); // 260
    int64_t numSubBlocks = qk / subBlock;             //   8
    int64_t quarter = 8;                              // 8-elem quarters
    int64_t numLanes = 8;                             // aux32/sums lanes
    int64_t numBsums = qk / 16;                       //  16

    // The integer-core vector types (shared with q4_K via the context) + the
    // fp32 fold types.
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    // q5_K reuses the q4_K core only at the mf2 default this increment, so the
    // canonical 8-lane type equals i32m2Type — passed directly into the cx
    // struct init below (no separate local needed; the wide fold-back branch is
    // unreached on the mf2/q5_K path).
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    mlir::Type u32PtrType = emitc::PointerType::get(constU32Type);
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constI16Type = emitc::OpaqueType::get(ctx, "const int16_t");
    mlir::Type constI16PtrType = emitc::PointerType::get(constI16Type);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination shared with the integer core).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // uint32_t utmp[4];  (function-scoped scratch for the bit-dance; the 16 bytes
    // utmp[0..3] are exactly [scales[0..7], mins[0..7]]).
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // float sums8[8];  (function-scoped scratch the 8-lane fp32 accumulator
    // vector is stored into before the SEQUENTIAL horizontal sum).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 positive accumulator, declared + zeroed ONCE OUTSIDE the super-block
    // loop (unlike aux32, which resets each super-block).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string sumsSeedCallee = "__riscv_vfmv_v_f_f32m2";
    mlir::Value sumsZero = emitOpaqueCallBuilt(
        rewriter, loc, f32m2Type, sumsSeedCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroF =
              rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
          return {zeroF, sizeLit(8)};
        });
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsZero);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator the MIN
    // subtractions land in IN-LOOP (in super-block order); declared + zeroed ONCE
    // OUTSIDE the loop.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // Per-super-block base address arithmetic (vx + ib*176, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The integer-core context shared with q4_K. The ONLY q5_K difference is the
    // qh 5th-bit plane: hasQh = true + the qh @16 offset. With hasQh set, the
    // shared core injects +16 per element whose qh bit is set (q5 in [0,31]);
    // every other emitted node is identical to q4_K's K4b.
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        activationPtrType, subBlock,  scalesOffset,  qsOffset,
        q8Offset,      numSubBlocks,  quarter,
        // q5_K reuses the q4_K core at the DEFAULT mf2 anchor (its op carries no
        // integer_core_lmul knob this increment) -- the Region-C chain stays
        // mf2 -> m1 -> m2, byte-identical to the legacy emit.
        /*coreLmul=*/"mf2", /*l8=*/"mf2", /*l16=*/"m1", /*l32=*/"m2",
        // S3-S4 strip params at the mf2 default: 8-wide strip, 4 strips/sub-block,
        // foldGroups=1 (no fold-back -- the legacy emitQuarter path runs). The
        // canonical-8 type is unused on the mf2 path (the wide fold-back branch is
        // unreached); pass i32m2Type (== canonical at mf2) to keep it non-null.
        /*stripWidth=*/8, /*numStrips=*/4, /*foldGroups=*/1,
        /*i32Canon8Type=*/i32m2Type,
        /*hasQh=*/true,
        /*qhOffset=*/qhOffset};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared q4_K/q5_K integer core: unpack (WITH the qh 5th-bit inject for
      // q5_K) + bit-dance + per-sub-block i32 dot, returning the per-super-block
      // aux32[8] lvalue AND the decoded scales/mins pointer (mins = scalesU8 + 8).
      // A null scaleMinOutput so the helper elides the K4a 16-byte scale/min
      // store.
      Q4_KCoreResult core = emitQ4_KSuperBlockAux32Core(
          rewriter, loc, cx, xb, yb, aux8Array, aux8Base, utmpArray,
          /*scaleMinOutput=*/nullptr, /*ib=*/nullptr);
      mlir::TypedValue<emitc::LValueType> aux32Var = core.aux32Var;
      mlir::Value scalesU8 = core.scalesU8;

      // The fp32 activation scale dy = *(const float *)(yb + 0), loaded ONCE and
      // reused by both the positive d and the min-term dmin.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_activation_d"));
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();

      // ---- the q4_K MIN term (reused node-for-node) ----
      // int sumi = 0; for (j=0..15) sumi += (int)bsums[j] * (int)mins[j/2];
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "min_term_bsums"));
      mlir::Value bsumsAddr = yb;
      if (bsumsOffset != 0)
        bsumsAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                  sizeLit(bsumsOffset));
      mlir::Value bsumsPtr =
          rewriter.create<emitc::CastOp>(loc, constI16PtrType, bsumsAddr)
              .getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value sumiZero =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
      rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiZero);
      for (int64_t j = 0; j < numBsums; ++j) {
        mlir::Value bIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(j));
        mlir::Value bElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(bsumsPtr),
                    bIdx)
                .getResult();
        mlir::Value bs16 =
            rewriter.create<emitc::LoadOp>(loc, constI16Type, bElem)
                .getResult();
        mlir::Value bs =
            rewriter.create<emitc::CastOp>(loc, i32Type, bs16).getResult();
        mlir::Value mIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(8 + j / 2));
        mlir::Value mElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(scalesU8),
                    mIdx)
                .getResult();
        mlir::Value mU8 =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, mElem).getResult();
        mlir::Value m =
            rewriter.create<emitc::CastOp>(loc, i32Type, mU8).getResult();
        mlir::Value prod =
            rewriter.create<emitc::MulOp>(loc, i32Type, bs, m).getResult();
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
        mlir::Value sumiNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumiCur, prod)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
      }

      // ---- the DEFERRED two-level fp32 positive fold (reused node-for-node) ----
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      mlir::Value aux32Val =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
      mlir::Value af = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, cvtCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {aux32Val, sizeLit(8)};
          });
      std::string mulCallee = "__riscv_vfmul_vf_f32m2";
      mlir::Value pr = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, mulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {af, d, sizeLit(8)};
          });
      std::string addCallee = "__riscv_vfadd_vv_f32m2";
      mlir::Value sumsNext = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, addCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumsCur =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar)
                    .getResult();
            return {sumsCur, pr, sizeLit(8)};
          });
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sums", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);

      // ---- the MIN subtraction: sumf -= dmin * (float)sumi (reused) ----
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_dmin"));
      mlir::Value dmAddr = xb;
      if (weightDminOffset != 0)
        dmAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDminOffset));
      mlir::Value dmx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                       mlir::ValueRange{dmAddr}, opName, role,
                                       llvm::StringRef("fcvt.s.h"));
      mlir::Value dmin =
          rewriter.create<emitc::MulOp>(loc, floatType, dmx, dy).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "min_subtract"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto minExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&minExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value minProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dmin, sumiFloat);
        mlir::Value sumfNext =
            rewriter.create<emitc::SubOp>(loc, floatType, sumfCur, minProduct);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, minExpr.getResult());
    }

    // ---- the SEQUENTIAL horizontal sum, l = 0 .. 7 (reused node-for-node) ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums_lanes"));
    mlir::Value sums8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sums8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sums8Array,
                                        mlir::ValueRange{sums8Index0})
            .getResult();
    mlir::Value sums8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(floatType), "&",
                                    sums8Elem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    emitVCallVoidBuilt(rewriter, loc, "vse32", "v_f32m2", opName, role,
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         return {sums8Base, sumsFinal, sizeLit(8)};
                       });

    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "horizontal_sum"));
    auto loadLane = [&](int64_t l) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(l));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(loc, sums8Array, mlir::ValueRange{idx})
              .getResult();
      return rewriter.create<emitc::LoadOp>(loc, floatType, elem).getResult();
    };
    mlir::Value sumf =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    for (int64_t l = 0; l < numLanes; ++l)
      sumf =
          rewriter.create<emitc::AddOp>(loc, floatType, sumf, loadLane(l))
              .getResult();

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q5_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ2_KQ8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ2KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ2KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q2_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q2_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 16
    int64_t weightStride = blockDot.getWeightBlockStride();         // 84
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();    //  0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            // 16
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        // 80
    int64_t weightDminOffset = blockDot.getWeightDminByteOffset();  // 82
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//  0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //  4
    int64_t bsumsOffset = blockDot.getActivationBsumsByteOffset();  // 260
    int64_t numSubBlocks = qk / subBlock;             // 16

    // The vector types for the 2-bit unpack (e8m2 32-lane chunk) and the
    // per-sub-block widening dot (e8m1 16-lane -> i16m2 -> i32m1). The dot uses
    // LMUL=1 (e8m1) so the 16-element sub-block reduce sees all 16 lanes
    // (VLMAX(e8m1) = VLEN/8 = 16 at VLEN >= 128; e8mf2 would cap at 8 and halve
    // the per-sub-block isuml).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8m1Type = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16m2Type = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constI16Type = emitc::OpaqueType::get(ctx, "const int16_t");
    mlir::Type constI16PtrType = emitc::PointerType::get(constI16Type);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered 2-bit
    // unpack destination read contiguously by the per-sub-block dot).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator the per-super-
    // block fold lands in IN-LOOP (in super-block order); declared + zeroed ONCE
    // OUTSIDE the loop (mirrors _generic's `float sumf = 0;`). q2_K has NO 8-lane
    // sums vector -- the positive term is the scalar isum.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // Per-super-block base address arithmetic (vx + ib*84, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // ---- (A) the 2-bit weight unpack into aux8[256] (element-ordered) ----
      // For each 32-byte qs chunk (chunk in 0..1) and each 2-bit shift in
      // {0,2,4,6}: aux8[128*chunk + 32*(shift/2) + l] = (qs[chunk*32+l] >> shift)
      // & 3 for the 32 lanes l. q2 in [0,3] is non-negative so the u8->i8
      // reinterpret is exact. Sub-block s -> aux8[16s:16s+16].
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "unpack_2bit"));
      std::string unpackSetvl = "__riscv_vsetvl_e8m2";
      for (int64_t chunk = 0; chunk < qk / 128; ++chunk) {
        int64_t qsChunk = chunk * 32; // q2 advances 32 bytes per 128-elem chunk
        mlir::Value vlu = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, unpackSetvl, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(32)};
            });
        mlir::Value qsPtr =
            byteOffsetPtr(xb, weightPtrType, qsOffset + qsChunk, u8PtrType);
        std::string loadCallee = "__riscv_vle8_v_u8m2";
        mlir::Value q2 =
            emitOpaqueCall(rewriter, loc, u8m2Type, loadCallee,
                           mlir::ValueRange{qsPtr, vlu}, opName, role);
        for (int64_t j = 0; j < 4; ++j) {
          int64_t shift = 2 * j;
          int64_t aChunk = chunk * 128 + j * 32; // aux8 base for this shift
          mlir::Value nib = q2;
          if (shift != 0) {
            std::string srlCallee = "__riscv_vsrl_vx_u8m2";
            nib = emitOpaqueCallBuilt(
                rewriter, loc, u8m2Type, srlCallee, opName, role,
                [&](mlir::OpBuilder &b,
                    mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                  mlir::Value amt = rewriter.create<emitc::LiteralOp>(
                      loc, i32ImmType, std::to_string(shift));
                  return {q2, amt, vlu};
                });
          }
          std::string andCallee = "__riscv_vand_vx_u8m2";
          mlir::Value q2bits = emitOpaqueCallBuilt(
              rewriter, loc, u8m2Type, andCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value mask3 =
                    rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0x03");
                return {nib, mask3, vlu};
              });
          std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
          mlir::Value q2i =
              emitOpaqueCall(rewriter, loc, i8m2Type, reCallee,
                             mlir::ValueRange{q2bits}, opName, role);
          mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
              loc, rewriter.getIndexType(), std::to_string(aChunk));
          mlir::Value dstElem =
              rewriter
                  .create<emitc::SubscriptOp>(loc, aux8Array,
                                              mlir::ValueRange{dstIdx})
                  .getResult();
          mlir::Value dstPtr =
              rewriter
                  .create<emitc::ApplyOp>(
                      loc, emitc::PointerType::get(i8ElemType), "&", dstElem)
                  .getResult();
          std::string storeCallee = "__riscv_vse8_v_i8m2";
          emitOpaqueCallVoid(rewriter, loc, storeCallee,
                             mlir::ValueRange{dstPtr, q2i, vlu}, opName, role);
        }
      }

      // The scales[16] base (the 16 direct packed 4-bit-scale/4-bit-min bytes).
      mlir::Value scBase =
          byteOffsetPtr(xb, weightPtrType, scalesOffset, u8PtrType);
      // The q8 quant base and the bsums base.
      mlir::Value q8Base =
          byteOffsetPtr(yb, activationPtrType, q8Offset, i8PtrType);
      mlir::Value bsumsAddr = yb;
      if (bsumsOffset != 0)
        bsumsAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                  sizeLit(bsumsOffset));
      mlir::Value bsumsPtr =
          rewriter.create<emitc::CastOp>(loc, constI16PtrType, bsumsAddr)
              .getResult();

      // ---- (B)/(C) the per-sub-block integer accumulation ----
      // int isum = 0;  (the per-super-block scalar positive accumulator).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("isum", opName, role));
      auto isumVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, isumVar,
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult());
      // int summs = 0;  (the per-super-block scalar min accumulator).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("summs", opName, role));
      auto summsVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, summsVar,
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult());

      for (int64_t s = 0; s < numSubBlocks; ++s) {
        // int sc = (int)scales[s];  (the packed scale/min byte).
        mlir::Value scIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(s));
        mlir::Value scElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(scBase),
                    scIdx)
                .getResult();
        mlir::Value scU8 =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, scElem).getResult();
        mlir::Value sc =
            rewriter.create<emitc::CastOp>(loc, i32Type, scU8).getResult();
        // scale = sc & 0xF  (the per-sub-block scale, the LOW nibble).
        mlir::Value scaleVal =
            rewriter
                .create<emitc::BitwiseAndOp>(
                    loc, i32Type, sc,
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0xF"))
                .getResult();
        // min = sc >> 4  (the per-sub-block min, the HIGH nibble).
        mlir::Value minVal =
            rewriter
                .create<emitc::BitwiseRightShiftOp>(
                    loc, i32Type, sc,
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "4"))
                .getResult();

        // isuml = Σ_{l=0..15} q8[16s+l] * aux8[16s+l]  (the vector widen-reduce;
        // integer / order-free). vle8 i8m1 (16 lanes) x2 -> vwmul_vv i16m2 ->
        // vwredsum_vs into i32m1 lane 0 (seed 0) -> vmv_x_s. LMUL=1 (e8m1) so the
        // 16-element reduce sees all 16 lanes (VLMAX(e8m1)=VLEN/8=16 at VLEN>=128;
        // e8mf2 would cap at 8 and halve the result).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_dot"));
        std::string dotSetvl = "__riscv_vsetvl_e8m1";
        mlir::Value vl16 = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, dotSetvl, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(subBlock)};
            });
        mlir::Value subOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, scIdx, sizeLit(subBlock));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, i8PtrType, q8Base, subOff)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, i8PtrType, aux8Base, subOff)
                .getResult();
        std::string loadCallee = "__riscv_vle8_v_i8m1";
        mlir::Value q8v =
            emitOpaqueCall(rewriter, loc, i8m1Type, loadCallee,
                           mlir::ValueRange{q8Ptr, vl16}, opName, role);
        mlir::Value av =
            emitOpaqueCall(rewriter, loc, i8m1Type, loadCallee,
                           mlir::ValueRange{aPtr, vl16}, opName, role);
        std::string mulCallee = "__riscv_vwmul_vv_i16m2";
        mlir::Value p =
            emitOpaqueCall(rewriter, loc, i16m2Type, mulCallee,
                           mlir::ValueRange{q8v, av, vl16}, opName, role);
        std::string seedCallee = "__riscv_vmv_v_x_i32m1";
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroImm =
                  rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0");
              return {zeroImm, sizeLit(1)};
            });
        std::string reduceCallee = "__riscv_vwredsum_vs_i16m2_i32m1";
        mlir::Value red =
            emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                           mlir::ValueRange{p, seed, vl16}, opName, role);
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value isuml =
            emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                           mlir::ValueRange{red}, opName, role);

        // isum += (sc & 0xF) * isuml;  (integer, order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "isum_accumulate"));
        mlir::Value scaleProd =
            rewriter.create<emitc::MulOp>(loc, i32Type, scaleVal, isuml)
                .getResult();
        mlir::Value isumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, isumVar).getResult();
        mlir::Value isumNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, isumCur, scaleProd)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, isumVar, isumNext);

        // summs += (int)bsums[s] * (sc >> 4);  (int16 sign-extended, order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "summs_accumulate"));
        mlir::Value bElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(bsumsPtr),
                    scIdx)
                .getResult();
        mlir::Value bs16 =
            rewriter.create<emitc::LoadOp>(loc, constI16Type, bElem).getResult();
        mlir::Value bs =
            rewriter.create<emitc::CastOp>(loc, i32Type, bs16).getResult();
        mlir::Value minProd =
            rewriter.create<emitc::MulOp>(loc, i32Type, bs, minVal).getResult();
        mlir::Value summsCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, summsVar).getResult();
        mlir::Value summsNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, summsCur, minProd)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, summsVar, summsNext);
      }

      // ---- (C) the SCALAR fp32 fold: sumf += dall*isum - dmin*summs ----
      // float dy = *(const float *)(yb + 0);  -- the fp32 activation scale,
      // loaded ONCE and shared by dall and dmin (mirrors _generic reading
      // `y[i].d` once).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_activation_d"));
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();

      // float dall = (float)*(const _Float16 *)(xb + 80) * dy;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_dall"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      mlir::Value dall =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // float dmin = (float)*(const _Float16 *)(xb + 82) * dy;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_dmin"));
      mlir::Value dmAddr = xb;
      if (weightDminOffset != 0)
        dmAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDminOffset));
      mlir::Value dmx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                       mlir::ValueRange{dmAddr}, opName, role,
                                       llvm::StringRef("fcvt.s.h"));
      mlir::Value dmin =
          rewriter.create<emitc::MulOp>(loc, floatType, dmx, dy).getResult();

      // sumf = sumf + (dall * (float)isum - dmin * (float)summs);  -- ONE
      // emitc.expression so the two products + the add + the subtract render as
      // ggml's single C statement (quants.c:561 `sumf += dall * isum - dmin *
      // summs`) and track its contraction. The emitc.load temps stay OUTSIDE.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scalar_fold"));
      mlir::Value isumFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, isumVar).getResult();
      mlir::Value summsFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, summsVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto foldExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&foldExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        // (float)isum, (float)summs  -- the int->float conversions.
        mlir::Value isumFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, isumFinal)
                .getResult();
        mlir::Value summsFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, summsFinal)
                .getResult();
        // dall * isum  -- the positive product.
        mlir::Value posProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dall, isumFloat);
        // dmin * summs  -- the min product.
        mlir::Value minProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dmin, summsFloat);
        // dall*isum - dmin*summs  -- the per-super-block delta.
        mlir::Value delta =
            rewriter.create<emitc::SubOp>(loc, floatType, posProduct,
                                          minProduct);
        // sumf + (dall*isum - dmin*summs)  -- the `+=`.
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, delta);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, foldExpr.getResult());
    }

    // *s = sumf;  (structured scalar store through the float * output pointer).
    mlir::Value sumf =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q2_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ3_KQ8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ3KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ3KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q3_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q3_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 16
    int64_t weightStride = blockDot.getWeightBlockStride();         // 110
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t hmaskOffset = blockDot.getWeightHmaskByteOffset();      //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();           //  32
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   //  96
    int64_t weightDOffset = blockDot.getWeightDByteOffset();       // 108
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             // 16
    int64_t half = subBlock / 2;                      //  8
    int64_t numLanes = numSubBlocks / 2;              //  8 (aux32/sums lanes)

    // The vector types: the 2-bit + hmask unpack (e8m2 32-lane chunk) and the
    // per-sub-block widening dot (e8mf2 8-lane -> i16m1 -> i32m2, the SAME q6_K
    // sub-block shape: e32m2 covers the 8 aux32 lanes at VLEN>=128).
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");

    // ---- the integer_core_lmul knob (mirrors q6_K) ----
    // The OPTIONAL knob anchors the per-sub-block integer-MAC chain i8->i16->i32.
    // mf2 (default, byte-identical to today): TWO 8-lane halves per 16-element
    // sub-block (i8mf2 -> i16m1 -> i32m2). m1 (ceiling): ONE 16-lane strip per
    // sub-block (i8m1 -> i16m2 -> i32m4), the 16 i32 lanes folded back to the
    // canonical 8 via a VLEN-AGNOSTIC vslidedown(literal offset 8)+vadd+vget(.,0)
    // BEFORE the fp32 cvt (the f32m2/8-lane fold is the byte-exact contract). The
    // 2-bit/subtractive-hmask unpack (e8m2) and signed 6-bit scale dance are
    // UNTOUCHED at every LMUL. Verifier pins coreLmul in {mf2,m1} (m2 rejected:
    // 16-element sub-block boundary). q3_K's emitter is inline (no shared helper),
    // so the knob feeds the SINGLE-SOURCE widening chain to locals here (the same
    // deriveWideningChain the q4_K/q6_K integer cores consume), replacing the
    // former inline per-callee {mf2,m1} ternaries -- byte-identical emit.
    llvm::StringRef coreLmul = blockDot.getIntegerCoreLmul().value_or("mf2");
    // l8/l16/l32 + stripWidth/foldGroups flow up the one widening chain. "m1" is
    // the q3_K hard ceiling (verifier rejects "m2"), so on this path the chain
    // only ever sees {mf2,m1}: mf2 -> i8mf2/i16m1/i32m2 (8-lane strip, no fold),
    // m1 -> i8m1/i16m2/i32m4 (16-lane strip, 2 fold groups).
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef l8 = wideningChain.l8;
    llvm::StringRef l16 = wideningChain.l16;
    llvm::StringRef l32 = wideningChain.l32;
    int64_t stripWidth = wideningChain.stripWidth;
    int64_t foldGroups = wideningChain.foldGroups;
    // The wide aux32 / MAC register-group types + callees follow l8/l16/l32.
    mlir::Type i8StripType =
        emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32WideType =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    std::string stripSetvlCallee = ("__riscv_vsetvl_e8" + l8).str();
    std::string stripLoadCallee = ("__riscv_vle8_v_i8" + l8).str();
    std::string mulWideCallee = ("__riscv_vwmul_vv_i16" + l16).str();
    std::string maccWideCallee = ("__riscv_vwmacc_vx_i32" + l32).str();
    std::string aux32SeedWideCallee = ("__riscv_vmv_v_x_i32" + l32).str();

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };
    // u8 immediate vector op: __riscv_<mnemonic>_u8m2(src, imm, vl).
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef imm, mlir::Value vl) -> mlir::Value {
      return emitVCallBuilt(
          rewriter, loc, u8m2Type, mnemonic, "u8m2", opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, i32ImmType, imm.str());
            return {src, amt, vl};
          });
    };
    auto u8VVOp = [&](llvm::StringRef mnemonic, mlir::Value a, mlir::Value b,
                      mlir::Value vl) -> mlir::Value {
      return emitVCall(rewriter, loc, u8m2Type, mnemonic, "u8m2",
                       mlir::ValueRange{a, b, vl}, opName, role);
    };
    auto u8Load = [&](mlir::Value ptr, mlir::Value vl) -> mlir::Value {
      return emitVCall(rewriter, loc, u8m2Type, "vle8_v", "u8m2",
                       mlir::ValueRange{ptr, vl}, opName, role);
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered 2-bit +
    // subtractive-hmask unpack destination read contiguously by the sub-loop).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // uint32_t utmp[4];  (the bit-dance output words; the 16 SIGNED 6-bit
    // scales are read as the int8 byte alias of these 4 words -- the same
    // type-pun _generic does with `const int8_t *scales = (const int8_t*)auxs`).
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // float sums8[8];  (the 8-lane fp32 accumulator is stored here before the
    // SEQUENTIAL horizontal sum -- identical to q6_K's sums8 scratch).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 accumulator, declared + zeroed ONCE OUTSIDE the loop (q6_K's `memset
    // (sums, 0, ...)` before the loop; aux32 resets each super-block).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string sumsSeedCallee = "__riscv_vfmv_v_f_f32m2";
    mlir::Value sumsZero = emitOpaqueCallBuilt(
        rewriter, loc, f32m2Type, sumsSeedCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroF =
              rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
          return {zeroF, sizeLit(8)};
        });
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsZero);

    // Per-super-block base address arithmetic (vx + ib*110, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // ---- (A) the 2-bit + SUBTRACTIVE-hmask unpack into aux8[256] ----
      // For each 32-byte qs chunk (chunk 0..1) and each 2-bit shift in {0,2,4,6}
      // (shiftIdx 0..3), with bit position p = 4*chunk + shiftIdx:
      //   aux8[128*chunk + 32*shiftIdx + l] =
      //     (((qs[chunk*32+l] >> shift) & 3) | (((hm[l] >> p) & 1) << 2)) - 4
      // The hmask plane is the SAME 32 bytes for ALL groups (`hm = x[i].hmask`
      // is NEVER advanced in _generic; only the tested bit `m = 1 << p` shifts).
      // Sub-block s -> aux8[16s:16s+16]. Result is SIGNED int8 in [-4,3].
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "unpack_2bit_subtractive_hmask"));
      std::string unpackSetvl = "__riscv_vsetvl_e8m2";
      for (int64_t chunk = 0; chunk < qk / 128; ++chunk) {
        int64_t qsChunk = chunk * 32; // qs advances 32 bytes per 128-elem chunk
        mlir::Value vlu = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, unpackSetvl, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(32)};
            });
        mlir::Value qsPtr =
            byteOffsetPtr(xb, weightPtrType, qsOffset + qsChunk, u8PtrType);
        mlir::Value qs = u8Load(qsPtr, vlu);
        // hm = x[i].hmask: the FIXED 32-byte plane, loaded once per chunk (the
        // e8m2 32-lane VLMAX matches the 32-element span; hm[l] is identical for
        // both chunks so reloading the same 32 bytes each chunk is exact).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "hmask_high_bit_plane"));
        mlir::Value hmPtr =
            byteOffsetPtr(xb, weightPtrType, hmaskOffset, u8PtrType);
        mlir::Value hm = u8Load(hmPtr, vlu);
        for (int64_t shiftIdx = 0; shiftIdx < 4; ++shiftIdx) {
          int64_t shift = 2 * shiftIdx;
          int64_t p = 4 * chunk + shiftIdx; // the hmask bit position
          int64_t aChunk = chunk * 128 + shiftIdx * 32; // aux8 base
          // low2 = (qs >> shift) & 3
          mlir::Value low = qs;
          if (shift != 0)
            low = u8ImmOp("vsrl_vx", qs, std::to_string(shift), vlu);
          mlir::Value low2 = u8ImmOp("vand_vx", low, "0x03", vlu);
          // hbit = ((hm >> p) & 1) << 2  (the high bit, lifted to bit-2)
          mlir::Value hshift =
              (p == 0) ? hm : u8ImmOp("vsrl_vx", hm, std::to_string(p), vlu);
          mlir::Value hbit = u8ImmOp("vand_vx", hshift, "0x01", vlu);
          mlir::Value hcontrib = u8ImmOp("vsll_vx", hbit, "0x02", vlu);
          // q3u = low2 | hcontrib  (in [0,7])
          mlir::Value q3u = u8VVOp("vor_vv", low2, hcontrib, vlu);
          // reinterpret to i8 (q3u in [0,7] non-negative so exact), then -4 ->
          // signed [-4,3]. This is _generic's `a = low2; a -= (hm&m ? 0 : 4)`:
          // bit set -> (low2|4)-4 = low2 in [0,3]; bit unset -> low2-4 in [-4,-1].
          std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
          mlir::Value q3i =
              emitOpaqueCall(rewriter, loc, i8m2Type, reCallee,
                             mlir::ValueRange{q3u}, opName, role);
          std::string subCallee = "__riscv_vsub_vx_i8m2";
          mlir::Value aVal = emitOpaqueCallBuilt(
              rewriter, loc, i8m2Type, subCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value bias4 =
                    rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "4");
                return {q3i, bias4, vlu};
              });
          mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
              loc, rewriter.getIndexType(), std::to_string(aChunk));
          mlir::Value dstElem =
              rewriter
                  .create<emitc::SubscriptOp>(loc, aux8Array,
                                              mlir::ValueRange{dstIdx})
                  .getResult();
          mlir::Value dstPtr =
              rewriter
                  .create<emitc::ApplyOp>(
                      loc, emitc::PointerType::get(i8ElemType), "&", dstElem)
                  .getResult();
          std::string storeCallee = "__riscv_vse8_v_i8m2";
          emitOpaqueCallVoid(rewriter, loc, storeCallee,
                             mlir::ValueRange{dstPtr, aVal, vlu}, opName, role);
        }
      }

      // ---- (B) the q3_K SIGNED 6-bit scale dance (STRUCTURED scalar emitc) ----
      // Read scales[12] as 3 u32 words w0/w1/w2; CAPTURE all three first (avoids
      // _generic's in-place read-before-write hazard), then reproduce _generic's
      // shuffle (quants.c:625-629, kmask1=0x03030303, kmask2=0x0f0f0f0f):
      //   tmp = w2;
      //   utmp[2] = ((w0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
      //   utmp[3] = ((w1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      //   utmp[0] = (w0 & kmask2)        | (((tmp >> 0) & kmask1) << 4);
      //   utmp[1] = (w1 & kmask2)        | (((tmp >> 2) & kmask1) << 4);
      // sc16 = (const int8_t *)utmp -> the 16 SIGNED 6-bit scales.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "signed_scale_bit_dance"));
      auto u32Lit = [&](llvm::StringRef v) -> mlir::Value {
        return rewriter.create<emitc::LiteralOp>(loc, u32Type, v.str());
      };
      mlir::Value kmask1 = u32Lit("0x03030303");
      mlir::Value kmask2 = u32Lit("0x0f0f0f0f");
      mlir::Value shift0 = u32Lit("0");
      mlir::Value shift2 = u32Lit("2");
      mlir::Value shift4 = u32Lit("4");
      mlir::Value shift6 = u32Lit("6");
      mlir::Value scWordPtr =
          byteOffsetPtr(xb, weightPtrType, scalesOffset, u32PtrType);
      auto loadWord = [&](int64_t idx) -> mlir::Value {
        mlir::Value wIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(idx));
        mlir::Value wElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(scWordPtr),
                    wIdx)
                .getResult();
        mlir::Value cw =
            rewriter.create<emitc::LoadOp>(loc, constU32Type, wElem)
                .getResult();
        return rewriter.create<emitc::CastOp>(loc, u32Type, cw).getResult();
      };
      mlir::Value w0 = loadWord(0);
      mlir::Value w1 = loadWord(1);
      mlir::Value w2 = loadWord(2); // tmp
      auto bAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseAndOp>(loc, u32Type, a, b)
            .getResult();
      };
      auto bOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseOrOp>(loc, u32Type, a, b)
            .getResult();
      };
      auto bShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseRightShiftOp>(loc, u32Type, a, b)
            .getResult();
      };
      auto bShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, u32Type, a, b)
            .getResult();
      };
      // utmp[0] = (w0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
      mlir::Value u0 =
          bOr(bAnd(w0, kmask2), bShl(bAnd(bShr(w2, shift0), kmask1), shift4));
      // utmp[1] = (w1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
      mlir::Value u1 =
          bOr(bAnd(w1, kmask2), bShl(bAnd(bShr(w2, shift2), kmask1), shift4));
      // utmp[2] = ((w0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
      mlir::Value u2 = bOr(bAnd(bShr(w0, shift4), kmask2),
                           bShl(bAnd(bShr(w2, shift4), kmask1), shift4));
      // utmp[3] = ((w1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
      mlir::Value u3 = bOr(bAnd(bShr(w1, shift4), kmask2),
                           bShl(bAnd(bShr(w2, shift6), kmask1), shift4));
      auto storeUtmp = [&](int64_t idx, mlir::Value v) {
        mlir::Value uIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(idx));
        mlir::Value uElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, utmpArray,
                                            mlir::ValueRange{uIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(uElem), v);
      };
      storeUtmp(0, u0);
      storeUtmp(1, u1);
      storeUtmp(2, u2);
      storeUtmp(3, u3);
      // const int8_t *sc = (const int8_t *)&utmp[0]  (the byte alias; the 16
      // 6-bit scales are read SIGNED).
      mlir::Value utmpIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value utmpElem0 =
          rewriter
              .create<emitc::SubscriptOp>(loc, utmpArray,
                                          mlir::ValueRange{utmpIndex0})
              .getResult();
      mlir::Value utmpWordPtr =
          rewriter
              .create<emitc::ApplyOp>(loc, emitc::PointerType::get(u32Type), "&",
                                      utmpElem0)
              .getResult();
      mlir::Value scI8Ptr =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, utmpWordPtr)
              .getResult();

      // ---- (B'/C) per-sub-block int8-scaled i32 dot into the aux32 strip ----
      // mf2: vint32m2_t aux32 = __riscv_vmv_v_x_i32m2(0, 8);  (8-lane, RESET/sblock)
      // m1 : vint32m4_t aux32 = __riscv_vmv_v_x_i32m4(0, 16); (16-lane wide strip)
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("aux32", opName, role));
      auto aux32Var = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32WideType),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value aux32Zero = emitOpaqueCallBuilt(
          rewriter, loc, i32WideType, aux32SeedWideCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zeroImm =
                rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0");
            return {zeroImm, sizeLit(stripWidth)};
          });
      rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Zero);

      // The q8 base for this super-block.
      mlir::Value q8Base =
          byteOffsetPtr(yb, activationPtrType, q8Offset, i8PtrType);

      // for (size_t js = 0; js < 16; js += 1) { ... }  (q6_K's sub-block loop;
      // the ONLY difference is the SIGNED scale scales[js]-32 instead of q6_K's
      // direct int8 scale).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "sub_block_loop"));
      auto subLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(numSubBlocks), sizeLit(1),
          /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard subGuard(rewriter);
        rewriter.setInsertionPointToStart(subLoop.getBody());
        mlir::Value js = subLoop.getInductionVar();

        // int scale = (int)sc[js] - 32;  (SIGNED int8 load, minus the q3_K bias).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "signed_scale_load"));
        mlir::Value scElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(scI8Ptr),
                    js)
                .getResult();
        mlir::Value scI8 =
            rewriter.create<emitc::LoadOp>(loc, constI8Type, scElem).getResult();
        mlir::Value scInt =
            rewriter.create<emitc::CastOp>(loc, i32Type, scI8).getResult();
        mlir::Value scale =
            rewriter
                .create<emitc::SubOp>(
                    loc, i32Type, scInt,
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "32"))
                .getResult();

        // js*16 -- the sub-block's first element offset into aux8 / q8.
        mlir::Value subBase =
            rewriter.create<emitc::MulOp>(loc, sizeType, js, sizeLit(subBlock));

        // One strip of `width` elements: vwmul i8xi8 -> i16, then vwmacc.vx the
        // wide aux32 += scale*i16. mf2 runs TWO 8-lane strips (== the legacy
        // emitHalf(0)/emitHalf(8), i8mf2/i16m1/i32m2); m1 runs ONE 16-lane strip
        // (i8m1/i16m2/i32m4) -- one 16-element sub-block under ONE scalar scale.
        // The per-strip provenance comment: mf2 keeps the legacy "sub_block_half"
        // tag so the default emit is BYTE-IDENTICAL to the committed kernel; m1's
        // single wide strip is tagged "sub_block_strip".
        std::string stripComment =
            (foldGroups > 1) ? "sub_block_strip" : "sub_block_half";
        auto emitStrip = [&](int64_t stripOffset) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, stripComment));
          mlir::Value vlStrip = emitOpaqueCallBuilt(
              rewriter, loc, sizeType, stripSetvlCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {sizeLit(stripWidth)};
              });
          mlir::Value off = subBase;
          if (stripOffset != 0)
            off = rewriter.create<emitc::AddOp>(loc, sizeType, subBase,
                                                sizeLit(stripOffset));
          mlir::Value q8Ptr =
              rewriter.create<emitc::AddOp>(loc, i8PtrType, q8Base, off)
                  .getResult();
          mlir::Value aPtr =
              rewriter.create<emitc::AddOp>(loc, i8PtrType, aux8Base, off)
                  .getResult();
          mlir::Value q8v =
              emitOpaqueCall(rewriter, loc, i8StripType, stripLoadCallee,
                             mlir::ValueRange{q8Ptr, vlStrip}, opName, role);
          mlir::Value av =
              emitOpaqueCall(rewriter, loc, i8StripType, stripLoadCallee,
                             mlir::ValueRange{aPtr, vlStrip}, opName, role);
          mlir::Value p =
              emitOpaqueCall(rewriter, loc, i16WideType, mulWideCallee,
                             mlir::ValueRange{q8v, av, vlStrip}, opName, role);
          mlir::Value aux32Next = emitOpaqueCallBuilt(
              rewriter, loc, i32WideType, maccWideCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value aux32Cur =
                    rewriter.create<emitc::LoadOp>(loc, i32WideType, aux32Var)
                        .getResult();
                return {aux32Cur, scale, p, vlStrip};
              });
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("aux32", opName, role));
          rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Next);
        };
        if (foldGroups > 1) {
          // m1: ONE 16-lane strip covers the whole sub-block.
          emitStrip(0);
        } else {
          // mf2 (default): TWO 8-lane halves (byte-identical to the legacy emit).
          emitStrip(0);
          emitStrip(half);
        }
      }

      // ---- (D) the NO-min DEFERRED two-level fp32 fold (q6_K's, identical) ----
      // float d = (float)*(const _Float16 *)(xb + 108) * *(const float *)(yb+0);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // Collapse the wide aux32 strip to the canonical 8-lane vint32m2_t BEFORE
      // the fp32 cvt (the f32m2/8-lane fold is the byte-exact contract).
      //   mf2 (foldGroups==1): load i32m2 DIRECTLY -- byte-identical to the legacy
      //                        emit (no fold tokens at all).
      //   m1  (foldGroups==2): VLEN-AGNOSTIC fold-back. The wide aux32 holds 16
      //     lanes (8 belonging to each 8-lane half); sum lane l with lane l+8.
      //     foldWide = aux32; foldWide = vadd(foldWide, vslidedown(foldWide, 8,
      //     vl=8), vl=8); aux32Val = vget_v_i32m4_i32m2(foldWide, 0). The slide
      //     offset is the LITERAL ELEMENT offset 8, so element 8+l lands at lane l
      //     at ANY VLEN, and vget(.,0) keeps the LOW 8 lanes (subgroup 0 is the low
      //     lanes at every VLEN). This is NOT a vget-subgroup extract (whose lane
      //     count = LMUL*VLEN/SEW is VLEN-dependent and broke q4_K at VLEN256) --
      //     it is the element-indexed vslidedown fix, correct at VLEN128 AND 256.
      //     aux32_8[l] = aux32_16[l] + aux32_16[l+8] is bit-identical to mf2's
      //     two-half accumulation (integer add is associative; each vwmacc stays
      //     within ONE 16-element sub-block under one scalar scale).
      mlir::Value aux32Val;
      if (foldGroups > 1) {
        mlir::Value foldWide =
            rewriter.create<emitc::LoadOp>(loc, i32WideType, aux32Var).getResult();
        std::string slideCallee = "__riscv_vslidedown_vx_i32m4";
        std::string addWideCallee = "__riscv_vadd_vv_i32m4";
        std::string getCallee = "__riscv_vget_v_i32m4_i32m2";
        for (int64_t g = 1; g < foldGroups; ++g) {
          mlir::Value slid = emitOpaqueCallBuilt(
              rewriter, loc, i32WideType, slideCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {foldWide, sizeLit(8 * g), sizeLit(8)};
              });
          foldWide = emitOpaqueCallBuilt(
              rewriter, loc, i32WideType, addWideCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                return {foldWide, slid, sizeLit(8)};
              });
        }
        aux32Val = emitOpaqueCallBuilt(
            rewriter, loc, i32m2Type, getCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value getIdx0 =
                  rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0");
              return {foldWide, getIdx0};
            });
      } else {
        // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);  -- mf2 loads its
        // 8-lane vint32m2_t aux32 unchanged (byte-identical to the legacy emit).
        aux32Val =
            rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      }
      std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
      mlir::Value af = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, cvtCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {aux32Val, sizeLit(8)};
          });
      // vfloat32m2_t pr = __riscv_vfmul_vf_f32m2(af, d, 8);  -- SEPARATE multiply
      // (NEVER a fused vfmacc).
      std::string mulCallee = "__riscv_vfmul_vf_f32m2";
      mlir::Value pr = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, mulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {af, d, sizeLit(8)};
          });
      // sums = __riscv_vfadd_vv_f32m2(sums, pr, 8);  -- SEPARATE add (NEVER fma).
      std::string addCallee = "__riscv_vfadd_vv_f32m2";
      mlir::Value sumsNext = emitOpaqueCallBuilt(
          rewriter, loc, f32m2Type, addCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumsCur =
                rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar)
                    .getResult();
            return {sumsCur, pr, sizeLit(8)};
          });
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sums", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);
    }

    // ---- (E) the SEQUENTIAL horizontal sum, l = 0 .. 7 (q6_K's, identical) ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums_lanes"));
    mlir::Value sums8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sums8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sums8Array,
                                        mlir::ValueRange{sums8Index0})
            .getResult();
    mlir::Value sums8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(floatType), "&",
                                    sums8Elem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    emitVCallVoidBuilt(rewriter, loc, "vse32", "v_f32m2", opName, role,
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         return {sums8Base, sumsFinal, sizeLit(8)};
                       });

    // float sumf = 0.0f; sumf += sums8[0]; ...; sumf += sums8[7];  -- the
    // SEQUENTIAL ascending fp32 horizontal sum (NEVER a vfredusum; fp add is
    // non-associative). Mirrors _generic's `float sumf = 0; for (l=0..7) sumf +=
    // sums[l]` INCLUDING the leading `0.0f +`.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto loadLane = [&](int64_t l) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(l));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(loc, sums8Array, mlir::ValueRange{idx})
              .getResult();
      return rewriter.create<emitc::LoadOp>(loc, floatType, elem).getResult();
    };
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "horizontal_sum"));
    mlir::Value sumf =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f").getResult();
    for (int64_t l = 0; l < numLanes; ++l)
      sumf =
          rewriter.create<emitc::AddOp>(loc, floatType, sumf, loadLane(l))
              .getResult();

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q3_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
