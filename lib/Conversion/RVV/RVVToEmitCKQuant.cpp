#include "RVVToEmitCInternal.h"
#include "Weft/Conversion/RVV/RVVToEmitCSupport.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace weft {
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

mlir::TypedValue<emitc::LValueType> VariantToEmitCFunc::emitQ3_KSuperBlockAux32Core(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q3_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base,
    mlir::TypedValue<emitc::ArrayType> utmpArray) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");

    // The per-sub-block integer-MAC widening chain follows the coreLmul knob (the
    // aux32 op stamps only mf2, so the constructed path only ever sees mf2 ->
    // i8mf2/i16m1/i32m2, 8-lane strip, no fold-back -- byte-identical to the
    // retired monolith's default emit). The unpack + signed scale dance are
    // LMUL-free at every knob.
    WideningChain wideningChain = deriveWideningChain(cx.coreLmul);
    llvm::StringRef l8 = wideningChain.l8;
    llvm::StringRef l16 = wideningChain.l16;
    llvm::StringRef l32 = wideningChain.l32;
    int64_t stripWidth = wideningChain.stripWidth;
    int64_t foldGroups = wideningChain.foldGroups;
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

    int64_t subBlock = cx.subBlock;             // 16
    int64_t numSubBlocks = cx.numSubBlocks;     // 16
    int64_t half = cx.half;                     //  8
    int64_t hmaskOffset = cx.hmaskOffset;       //  0
    int64_t qsOffset = cx.qsOffset;             // 32
    int64_t scalesOffset = cx.scalesOffset;     // 96
    int64_t q8Offset = cx.q8Offset;             //  4
    int64_t qk = subBlock * numSubBlocks;       // 256

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };
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

    // ---- (A) the 2-bit + SUBTRACTIVE-hmask unpack into aux8[256] ----
    // For each 32-byte qs chunk (chunk 0..1) and each 2-bit shift in {0,2,4,6}
    // (shiftIdx 0..3), with bit position p = 4*chunk + shiftIdx:
    //   aux8[128*chunk + 32*shiftIdx + l] =
    //     (((qs[chunk*32+l] >> shift) & 3) | (((hm[l] >> p) & 1) << 2)) - 4
    // The hmask plane is the SAME 32 bytes for ALL groups (`hm = x[i].hmask` is
    // NEVER advanced in _generic; only the tested bit `m = 1 << p` shifts). Sub-
    // block s -> aux8[16s:16s+16]. Result is SIGNED int8 in [-4,3].
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "unpack_2bit_subtractive_hmask"));
    std::string unpackSetvl = "__riscv_vsetvl_e8m2";
    for (int64_t chunk = 0; chunk < qk / 128; ++chunk) {
      int64_t qsChunk = chunk * 32; // qs advances 32 bytes per 128-elem chunk
      mlir::Value vlu = emitOpaqueCallBuilt(
          rewriter, loc, cx.sizeType, unpackSetvl, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(32)};
          });
      mlir::Value qsPtr =
          byteOffsetPtr(xb, cx.weightPtrType, qsOffset + qsChunk, u8PtrType);
      mlir::Value qs = u8Load(qsPtr, vlu);
      // hm = x[i].hmask: the FIXED 32-byte plane, loaded once per chunk.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "hmask_high_bit_plane"));
      mlir::Value hmPtr =
          byteOffsetPtr(xb, cx.weightPtrType, hmaskOffset, u8PtrType);
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
        // signed [-4,3]. This is _generic's `a = low2; a -= (hm&m ? 0 : 4)`.
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
    // shuffle (kmask1=0x03030303, kmask2=0x0f0f0f0f) into utmp[4]; the 16 SIGNED
    // 6-bit scales are read as the int8 byte alias of those 4 words.
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
        byteOffsetPtr(xb, cx.weightPtrType, scalesOffset, u32PtrType);
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
          rewriter.create<emitc::LoadOp>(loc, constU32Type, wElem).getResult();
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
      return rewriter.create<emitc::BitwiseOrOp>(loc, u32Type, a, b).getResult();
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
        rewriter.create<emitc::CastOp>(loc, i8PtrType, utmpWordPtr).getResult();

    // ---- (B'/C) per-sub-block int8-scaled i32 dot into the aux32 strip ----
    // mf2: vint32m2_t aux32 = __riscv_vmv_v_x_i32m2(0, 8);  (8-lane, RESET/sblock)
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
        byteOffsetPtr(yb, cx.activationPtrType, q8Offset, i8PtrType);

    // for (size_t js = 0; js < 16; js += 1) { ... }  (the ONLY difference from
    // q6_K is the SIGNED scale scales[js]-32 instead of q6_K's direct int8).
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
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(scI8Ptr), js)
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
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, js, sizeLit(subBlock));

      std::string stripComment =
          (foldGroups > 1) ? "sub_block_strip" : "sub_block_half";
      auto emitStrip = [&](int64_t stripOffset) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, stripComment));
        mlir::Value vlStrip = emitOpaqueCallBuilt(
            rewriter, loc, cx.sizeType, stripSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(stripWidth)};
            });
        mlir::Value off = subBase;
        if (stripOffset != 0)
          off = rewriter.create<emitc::AddOp>(loc, cx.sizeType, subBase,
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

    // ---- fold-back: collapse the wide aux32 to the canonical 8 lanes ----
    // Only emitted for foldGroups > 1 (m1). At mf2 (foldGroups==1) this block is
    // skipped entirely -> the returned aux32 IS the 8-lane vint32m2, byte-identical
    // to the retired monolith's default emit (the caller loads i32m2 directly).
    mlir::TypedValue<emitc::LValueType> resultLValue =
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(aux32Var.getResult());
    if (foldGroups > 1) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "aux32_fold_back"));
      mlir::Value foldWide =
          rewriter.create<emitc::LoadOp>(loc, i32WideType, aux32Var).getResult();
      for (int64_t g = 1; g < foldGroups; ++g) {
        std::string slideCallee = ("__riscv_vslidedown_vx_i32" + l32).str();
        mlir::Value slid = emitOpaqueCallBuilt(
            rewriter, loc, i32WideType, slideCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value slideOff = rewriter.create<emitc::LiteralOp>(
                  loc, i32ImmType, std::to_string(8 * g));
              return {foldWide, slideOff, sizeLit(8)};
            });
        std::string addCallee = ("__riscv_vadd_vv_i32" + l32).str();
        foldWide = emitOpaqueCallBuilt(
            rewriter, loc, i32WideType, addCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {foldWide, slid, sizeLit(8)};
            });
      }
      std::string getCallee = ("__riscv_vget_v_i32" + l32 + "_i32m2").str();
      mlir::Value foldCanon = emitOpaqueCallBuilt(
          rewriter, loc, i32m2Type, getCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zeroIdx =
                rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0");
            return {foldWide, zeroIdx};
          });
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("aux32_fold", opName, role));
      auto foldVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32m2Type),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(loc, foldVar, foldCanon);
      resultLValue =
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(foldVar.getResult());
    }

    return resultLValue;
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ6_KQ8_KAux32Partial(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlBlockDotQ6KQ8KAux32Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<weftrvv::GgmlBlockDotQ6KQ8KAux32Op>(op))
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

    llvm::StringRef opName = blockDot.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getWEFTEmitCLowerableSourceRole();
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

// Track B q4_K BRICK 1: the plain 4-bit nibble unpack into aux8[256] (Region A),
// factored out VERBATIM from emitQ4_KSuperBlockAux32Core so the SAME node
// sequence is reachable both inline (the monolithic q4_K/q5_K integer core) AND
// through the first-class weft_rvv.q4_k_nibble_unpack op. Emits at the current
// insertion point; writes aux8Array as a side effect.
void VariantToEmitCFunc::emitQ4_KPlainNibbleUnpack(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KIntegerCoreContext &cx, mlir::Value xb,
    mlir::TypedValue<emitc::ArrayType> aux8Array) const {
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    int64_t qk = cx.subBlock * cx.numSubBlocks; // 256
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
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
}

// Track B q4_K BRICK 2: the 6-bit scale/min bit-dance (utmp/kmask cross-byte
// decode into the 16 [scales,mins] bytes -- Region B), factored out VERBATIM
// from emitQ4_KSuperBlockAux32Core so the SAME node sequence is reachable both
// inline (the monolithic q4_K/q5_K integer core) AND through the first-class
// weft_rvv.q4_k_scale_min_bit_dance op. Emits at the current insertion point;
// fills utmpArray as a side effect and returns scalesU8 = (const uint8_t
// *)&utmp[0] (the 8 6-bit scales, contiguous with the 8 mins -- 16 bytes total).
mlir::Value VariantToEmitCFunc::emitQ4_KScaleMinBitDanceCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KIntegerCoreContext &cx, mlir::Value xb,
    mlir::TypedValue<emitc::ArrayType> utmpArray) const {
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
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
    return scalesU8;
}

// Track B q4_K BRICK 3: the per-sub-block uint6-scaled i32 dot into the 8-lane
// aux32 PLUS the integer fold-back (Region C), factored out VERBATIM from
// emitQ4_KSuperBlockAux32Core so the SAME node sequence is reachable both inline
// (the monolithic q4_K/q5_K integer core) AND through the first-class
// weft_rvv.q4_k_scaled_dot op. Emits at the current insertion point: declares the
// per-super-block aux32 lvalue, seeds it to zero, runs the sub-block loop applying
// the per-sub-block UINT6 scale `scalesU8[js]` in the i32 domain via the
// vsetvl_e8<l8>/vle8x2/vwmul i16<l16>/vwmacc i32<l32> MAC strip (cx.numStrips per
// 32-element sub-block), then (when foldGroups > 1) the VLEN-agnostic integer
// fold-back of the WIDE aux32's group-of-8 residues to the canonical 8-lane
// vint32m2_t accumulator (vslidedown + vadd register-only regroup + vget). The
// scale is FUSED into the vwmacc -- this whole MAC+fold region reads scalesU8 as
// ONE unit. Returns the canonical-8 aux32 lvalue (the byte-exact Region-F
// contract type). The scale_load reads scalesU8, the activation q8 base is
// derived from yb via cx.q8Offset/cx.activationPtrType, the aux8 base is
// aux8Base. The integer add is associative/order-free so this regroup is
// provably bit-exact at every legal LMUL (no fp non-associativity here -- that is
// Region F, deferred).
mlir::TypedValue<emitc::LValueType>
VariantToEmitCFunc::emitQ4_KScaledDotIntoAux32(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KIntegerCoreContext &cx, mlir::Value yb, mlir::Value aux8Base,
    mlir::Value scalesU8) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };

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

    return resultAux32Var;
}

// Track B q4_K BRICK 4 (first half): the int16 bsums load + the SCALAR integer
// reduction sumi = sum_j(bsums[j] * mins[j/2]), factored out VERBATIM from
// emitQ4_KQ8_KBlockDot so the SAME node sequence is reachable both inline (the
// monolithic q4_K/q5_K block dot, which emits it BEFORE its deferred positive
// fold) AND through the first-class weft_rvv.q4_k_min_term op. Emits at the
// current insertion point: the bsums base (yb + bsumsOffset, cast const int16_t
// *), declares + zeroes the per-super-block scalar `int sumi`, then the
// numBsums-iteration scalar reduction pairing each SIGN-extended int16 bsums with
// the decoded uint6 min scalesU8[8 + j/2] (mins = scalesU8 + 8; each min spans
// TWO consecutive bsums). The integer multiply/add is associative/order-free so
// this is byte-exact at any reduction tree. Returns the scalar sumi lvalue. The
// activation pointer type is derived from yb; the decoded scales/mins base is
// scalesU8.
mlir::TypedValue<emitc::LValueType> VariantToEmitCFunc::emitQ4_KMinTermBsumsDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KMinTermContext &mx, mlir::Value yb, mlir::Value scalesU8) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = mx.opName;
    llvm::StringRef role = mx.role;
    mlir::Type activationPtrType = yb.getType();
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, mx.sizeType,
                                               std::to_string(v));
    };

    // const int16_t *bsums = (const int16_t *)(yb + 260);  -- the q8_K block
    // bsums, int16 (SIGN-extended on load).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "min_term_bsums"));
    mlir::Value bsumsAddr = yb;
    if (mx.bsumsOffset != 0)
      bsumsAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(mx.bsumsOffset));
    mlir::Value bsumsPtr =
        rewriter.create<emitc::CastOp>(loc, mx.constI16PtrType, bsumsAddr)
            .getResult();
    // int sumi = 0;  (the per-super-block scalar min accumulator).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumi", opName, role));
    auto sumiVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(mx.i32Type), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumiZero =
        rewriter.create<emitc::LiteralOp>(loc, mx.i32Type, "0");
    rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiZero);
    for (int64_t j = 0; j < mx.numBsums; ++j) {
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
          rewriter.create<emitc::LoadOp>(loc, mx.constI16Type, bElem)
              .getResult();
      mlir::Value bs =
          rewriter.create<emitc::CastOp>(loc, mx.i32Type, bs16).getResult();
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
          rewriter.create<emitc::LoadOp>(loc, mx.constU8Type, mElem).getResult();
      mlir::Value m =
          rewriter.create<emitc::CastOp>(loc, mx.i32Type, mU8).getResult();
      // sumi += bsums[j] * mins[j/2];  (integer, order-free).
      mlir::Value prod =
          rewriter.create<emitc::MulOp>(loc, mx.i32Type, bs, m).getResult();
      mlir::Value sumiCur =
          rewriter.create<emitc::LoadOp>(loc, mx.i32Type, sumiVar).getResult();
      mlir::Value sumiNext =
          rewriter.create<emitc::AddOp>(loc, mx.i32Type, sumiCur, prod)
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
    }
    return llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumiVar.getResult());
}

// Track B q4_K BRICK 4 (second half): the fp16 dmin read + the single fp
// contraction sumf -= dmin * (float)sumi, factored out VERBATIM from
// emitQ4_KQ8_KBlockDot so the SAME node sequence is reachable both inline (the
// monolithic q4_K/q5_K block dot, which emits it AFTER its positive fold) AND
// through the first-class weft_rvv.q4_k_min_term op. Emits at the current
// insertion point: dmin = fp16(*(const _Float16 *)(xb + weightDminOffset)) * dy
// (the once-loaded fp32 activation scale), then `sumf = sumf - dmin *
// (float)sumi` as ONE emitc.expression (ggml's `sumf -= dmin * sumi`,
// quants.c:714, the emitc.load temps OUTSIDE the expression), then the AssignOp
// into the carried sumf lvalue. The final scalar fp multiply is fixed-order --
// no fp-reassociation seam. The weight pointer type is derived from xb.
void VariantToEmitCFunc::emitQ4_KMinTermSubtract(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KMinTermContext &mx, mlir::Value xb, mlir::Value dy,
    mlir::TypedValue<emitc::LValueType> sumiVar,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    llvm::StringRef opName = mx.opName;
    llvm::StringRef role = mx.role;
    mlir::Type weightPtrType = xb.getType();
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, mx.sizeType,
                                               std::to_string(v));
    };

    // float dmin = (float)*(const _Float16 *)(xb + 2) * dy;  -- the fp16 weight
    // min scale (byte 2) times the SAME fp32 activation scale.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "fold_scale_dmin"));
    mlir::Value dmAddr = xb;
    if (mx.weightDminOffset != 0)
      dmAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                             sizeLit(mx.weightDminOffset));
    mlir::Value dmx =
        emitOpaqueCall(rewriter, loc, mx.floatType, mx.fp16ReadCallee,
                       mlir::ValueRange{dmAddr}, opName, role,
                       llvm::StringRef("fcvt.s.h"));
    mlir::Value dmin =
        rewriter.create<emitc::MulOp>(loc, mx.floatType, dmx, dy).getResult();
    // sumf = sumf - dmin * (float)sumi;  -- ONE emitc.expression so it renders
    // as ggml's single C statement (quants.c:714 `sumf -= dmin * sumi`) and
    // tracks its contraction. The emitc.load temps stay OUTSIDE the expression.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "min_subtract"));
    mlir::Value sumiFinal =
        rewriter.create<emitc::LoadOp>(loc, mx.i32Type, sumiVar).getResult();
    mlir::Value sumfCur =
        rewriter.create<emitc::LoadOp>(loc, mx.floatType, sumfVar).getResult();
    auto minExpr = rewriter.create<emitc::ExpressionOp>(
        loc, mx.floatType, /*do_not_inline=*/false);
    {
      mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
      mlir::Block *exprBlock = rewriter.createBlock(&minExpr.getRegion());
      rewriter.setInsertionPointToStart(exprBlock);
      mlir::Value sumiFloat =
          rewriter.create<emitc::CastOp>(loc, mx.floatType, sumiFinal)
              .getResult();
      // dmin * (float)sumi  -- the min product (ggml binds `dmin * sumi`
      // before the `-=`).
      mlir::Value minProduct =
          rewriter.create<emitc::MulOp>(loc, mx.floatType, dmin, sumiFloat);
      // sumf - (dmin * sumi)  -- the `-=` subtraction tree.
      mlir::Value sumfNext =
          rewriter.create<emitc::SubOp>(loc, mx.floatType, sumfCur, minProduct);
      rewriter.create<emitc::YieldOp>(loc, sumfNext);
    }
    rewriter.create<emitc::VerbatimOp>(loc, assignComment("sumf", opName, role));
    rewriter.create<emitc::AssignOp>(loc, sumfVar, minExpr.getResult());
}

// Track B q4_K BRICK 6: the DEFERRED two-level fp32 POSITIVE fold (q6_K K2
// mechanism), `sums += (fp16(x.d) * dy) * (float)aux32`, factored out VERBATIM
// from emitQ4_KQ8_KBlockDot so the SAME node sequence is reachable both inline
// (the monolithic q4_K/q5_K block dot, which emits it BETWEEN its two interleaved
// MIN-term halves) AND through the first-class weft_rvv.q4_k_sums_fold_scale_d
// op. Emits at the current insertion point: d = fp16(*(const _Float16 *)(xb +
// weightDOffset)) * dy (the once-loaded fp32 activation scale), af = vfcvt of the
// BRICK 3 canonical-8 aux32, pr = vfmul.vf(af, d) (a SEPARATE multiply, NEVER an
// fma), sums = vfadd.vv(sums, pr) (a SEPARATE add), then the AssignOp into the
// carried sums lvalue. The per-sub-block uint6 scale is NOT read here (BRICK 3
// fused it into aux32), and BRICK 3's integer fold-back already collapsed aux32
// to canonical-8 vint32m2_t -- so this fold is a fixed-order 8-lane sequence with
// NO fp-reassociation seam. The weight pointer type is derived from xb.
void VariantToEmitCFunc::emitQ4_KSumsFoldScaleD(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KSumsFoldContext &sx, mlir::Value xb, mlir::Value dy,
    mlir::TypedValue<emitc::LValueType> aux32Var,
    mlir::TypedValue<emitc::LValueType> sumsVar) const {
    llvm::StringRef opName = sx.opName;
    llvm::StringRef role = sx.role;
    mlir::Type weightPtrType = xb.getType();
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sx.sizeType,
                                               std::to_string(v));
    };

    // float d = (float)*(const _Float16 *)(xb + 0) * dy;  -- the fp16 weight
    // super-block scale (byte 0) times the fp32 activation scale (loaded once).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "fold_scale_d"));
    mlir::Value dxAddr = xb;
    if (sx.weightDOffset != 0)
      dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                             sizeLit(sx.weightDOffset));
    mlir::Value dx =
        emitOpaqueCall(rewriter, loc, sx.floatType, sx.fp16ReadCallee,
                       mlir::ValueRange{dxAddr}, opName, role,
                       llvm::StringRef("fcvt.s.h"));
    mlir::Value d =
        rewriter.create<emitc::MulOp>(loc, sx.floatType, dx, dy).getResult();

    // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
    // The aux32 lvalue is the CANONICAL 8-lane vint32m2 (the S4 fold-back collapses
    // the wide aux32 at m1/m2; at mf2/q5_K it is the running aux32, same type) --
    // load it as i32Canon8Type (always vint32m2_t), byte-identical at every anchor.
    mlir::Value aux32Val =
        rewriter.create<emitc::LoadOp>(loc, sx.i32Canon8Type, aux32Var)
            .getResult();
    std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
    mlir::Value af = emitOpaqueCallBuilt(
        rewriter, loc, sx.f32m2Type, cvtCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {aux32Val, sizeLit(8)};
        });
    // vfloat32m2_t pr = __riscv_vfmul_vf_f32m2(af, d, 8);  -- SEPARATE multiply.
    std::string mulCallee = "__riscv_vfmul_vf_f32m2";
    mlir::Value pr = emitOpaqueCallBuilt(
        rewriter, loc, sx.f32m2Type, mulCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {af, d, sizeLit(8)};
        });
    // sums = __riscv_vfadd_vv_f32m2(sums, pr, 8);  -- SEPARATE add (NEVER fma).
    std::string addCallee = "__riscv_vfadd_vv_f32m2";
    mlir::Value sumsNext = emitOpaqueCallBuilt(
        rewriter, loc, sx.f32m2Type, addCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value sumsCur =
              rewriter.create<emitc::LoadOp>(loc, sx.f32m2Type, sumsVar)
                  .getResult();
          return {sumsCur, pr, sizeLit(8)};
        });
    rewriter.create<emitc::VerbatimOp>(loc, assignComment("sums", opName, role));
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);
}

// Track B q4_K BRICK 7: the POST-LOOP horizontal fold, factored out VERBATIM from
// emitQ4_KQ8_KBlockDot so the SAME node sequence is reachable both inline (the
// monolithic q4_K block dot, which emits it AFTER its super-block loop) AND through
// the first-class weft_rvv.q4_k_horizontal_fold op. Emits at the current insertion
// point: `vse32_v_f32m2(&sums8[0], sums, 8)` materializes the carried 8-lane fp32
// sums accumulator into the sums8[8] scratch, then the FIXED sequential ascending
// `sumf = sumf + sums8[l]` for l = 0..numLanes-1 (numLanes == 8) collapses the 8
// lanes into the carried scalar sumf (which already holds the in-loop MIN
// subtractions). The horizontal sum is anchor-INDEPENDENT (always 8 lanes, fixed
// ascending order, NEVER a vfredusum) -- it mirrors _generic's `for (l=0..7) sumf
// += sums[l]`, so there is NO fp-reassociation seam here. Does NOT emit the final
// store (the monolith stores `*s = sumf` through the ABI float *, the standalone op
// stores into a local sink) -- RETURNS the final sumf Value so each caller can store
// it.
mlir::Value VariantToEmitCFunc::emitQ4_KHorizontalFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q4_KHorizontalFoldContext &hx,
    mlir::TypedValue<emitc::ArrayType> sums8Array,
    mlir::TypedValue<emitc::LValueType> sumsVar,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    llvm::StringRef opName = hx.opName;
    llvm::StringRef role = hx.role;
    mlir::Type floatType = hx.floatType;
    mlir::Type f32m2Type = hx.f32m2Type;
    int64_t numLanes = hx.numLanes;
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, hx.sizeType,
                                               std::to_string(v));
    };

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
    return sumf;
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
    // NB: the legacy `quarter` local (== cx.quarter == 8) is gone — the strip
    // offset is now derived from cx.stripWidth (mf2 -> 8 == the old quarter).
    // The super-block element count `qk` (== cx.subBlock * cx.numSubBlocks) is
    // now consumed inside the Region-A unpack helper, not here.

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    // NB: the byteOffsetPtr lambda that used to live here is now inside the
    // shared Track B brick-3 helper emitQ4_KScaledDotIntoAux32 (the only Region
    // that used it -- the q8 base derivation); sizeLit stays for the K4a-only
    // store_scale_min vse8 below.

    // ---- (A) 4-bit nibble unpack into aux8 (element-ordered, NO bias) ----
    // Factored into the shared Track B brick-1 helper (the SAME node sequence,
    // also reachable through the first-class weft_rvv.q4_k_nibble_unpack op).
    emitQ4_KPlainNibbleUnpack(rewriter, loc, cx, xb, aux8Array);

    // ---- (B) the 6-bit scale/min bit-dance (utmp/kmask), STRUCTURED scalar
    // emitc.bitwise ops (NO raw string), mirroring _generic (quants.c:685-690).
    // Factored into the shared Track B brick-2 helper (the SAME node sequence,
    // also reachable through the first-class weft_rvv.q4_k_scale_min_bit_dance
    // op); returns scalesU8 = (const uint8_t *)&utmp[0].
    mlir::Value scalesU8 =
        emitQ4_KScaleMinBitDanceCore(rewriter, loc, cx, xb, utmpArray);

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

    // ---- (C) per-sub-block uint6-scaled i32 dot into the 8-lane aux32 + the
    // C-tail integer fold-back ----
    // Factored into the shared Track B brick-3 helper (the SAME node sequence,
    // also reachable through the first-class weft_rvv.q4_k_scaled_dot op); it
    // seeds + accumulates the running (wide) aux32 via the integer_core_lmul MAC
    // chain reading scalesU8/aux8Base and the q8 base derived from yb, then folds
    // the wide aux32 back to the canonical 8-lane vint32m2 (no-op at mf2), and
    // returns the canonical-8 aux32 lvalue Region F consumes.
    mlir::TypedValue<emitc::LValueType> resultAux32Var =
        emitQ4_KScaledDotIntoAux32(rewriter, loc, cx, yb, aux8Base, scalesU8);

    return Q4_KCoreResult{resultAux32Var, scalesU8};
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KQ8_KAux32Partial(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::GgmlBlockDotQ4KQ8KAux32Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<weftrvv::GgmlBlockDotQ4KQ8KAux32Op>(op))
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

    llvm::StringRef opName = blockDot.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getWEFTEmitCLowerableSourceRole();
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

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KNibbleUnpack(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    (void)avlArg; // Region A decodes ONE super-block at a literal vl; no n use.
    weftrvv::Q4KNibbleUnpackOp unpack;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto u = llvm::dyn_cast<weftrvv::Q4KNibbleUnpackOp>(op))
        unpack = u;
    }
    if (!unpack)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K nibble-unpack body missing the op");

    mlir::Value weightBase = valueMap.lookup(unpack.getWeightBase());
    if (!weightBase)
      return rewriter.notifyMatchFailure(unpack,
                                         "q4_K nibble-unpack weight base unmapped");

    llvm::StringRef opName = unpack.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = unpack.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();

    // The integer-core vector types Region A uses (the 4-bit unpack runs 32-wide
    // chunks at e8m2; VLMAX = 32 at VLEN >= 128). The non-Region-A cx type/offset
    // fields are populated for a total context but are unused by the unpack.
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
    mlir::Type u32PtrType = emitc::PointerType::get(constU32Type);

    // The Region-A structural facts come straight off the typed attrs (I4).
    int64_t qk = unpack.getQk();                  // 256
    int64_t subBlock = unpack.getSubBlock();      // 32
    int64_t qsOffset = unpack.getWeightQsByteOffset(); // 16
    int64_t numSubBlocks = qk / subBlock;         //  8

    // The integer-core context: the SAME shape the monolithic q4_K core builds.
    // Region A reads opName/role, subBlock/numSubBlocks (for qk), qsOffset,
    // sizeType, weightPtrType, u8PtrType, u8m2Type, i32ImmType, i8m2Type,
    // i8ElemType, hasQh(default false). The remaining fields complete the
    // context and are unused by the unpack (no activation, no scales, no dot).
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        /*activationPtrType=*/weightPtrType, subBlock, /*scalesOffset=*/0,
        qsOffset,      /*q8Offset=*/0, numSubBlocks,  /*quarter=*/8};

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // int8_t aux8[256];  (function-scoped scratch the Region-A unpack fills; the
    // op DECLARES it so the brick is self-contained, vs the monolithic core which
    // declares it once above the super-block loop and shares it with the dot).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());

    // The weight base operand IS this op's super-block pointer (single super-block
    // -- no nb = n/256 loop, no ib*144 address arithmetic). Region A's emit is the
    // SAME node sequence the monolithic core emits inside its loop.
    emitQ4_KPlainNibbleUnpack(rewriter, loc, cx, weightBase, aux8Array);

    // The op's i32 m1 result token: the unpack writes aux8 as a side effect, so
    // the token has no live use; bind it to a zero literal to keep the map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[unpack.getResult()] = resultToken;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KScaleMinBitDance(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    (void)avlArg; // Region B decodes ONE super-block; the bit-dance is pure
                  // scalar and the vse8 observable uses a literal vl -- no n use.
    weftrvv::Q4KScaleMinBitDanceOp bitDance;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<weftrvv::Q4KScaleMinBitDanceOp>(op))
        bitDance = bd;
    }
    if (!bitDance)
      return rewriter.notifyMatchFailure(
          scope, "q4_K scale-min-bit-dance body missing the op");

    mlir::Value weightBase = valueMap.lookup(bitDance.getWeightBase());
    if (!weightBase)
      return rewriter.notifyMatchFailure(
          bitDance, "q4_K scale-min-bit-dance weight base unmapped");

    llvm::StringRef opName = bitDance.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = bitDance.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();

    // The integer-core vector types Region B's context carries. The bit-dance
    // itself is pure scalar (uint32 word loads + emitc.bitwise into utmp); the
    // vector type fields are populated for a total context but unused by it.
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
    mlir::Type u32PtrType = emitc::PointerType::get(constU32Type);

    // The Region-B structural facts come straight off the typed attrs (I4).
    int64_t qk = bitDance.getQk();                          // 256
    int64_t subBlock = bitDance.getSubBlock();              // 32
    int64_t scalesOffset = bitDance.getWeightScalesByteOffset(); // 4
    int64_t numSubBlocks = qk / subBlock;                   //  8

    // The integer-core context: the SAME shape the monolithic q4_K core builds.
    // Region B reads opName/role, sizeType, u32Type, weightPtrType, scalesOffset,
    // u32PtrType, constU32Type, u8PtrType. The remaining fields complete the
    // context and are unused by the bit-dance (no activation, no nibble, no dot).
    // DESIGNATED (explicit field assignment, C++17) rather than positional
    // aggregate-init: emitQ4_KScaleMinBitDanceCore does not read the Region-C
    // l8/l16/l32/stripWidth fields, but a future mid-struct field insertion would
    // silently misfeed a positional init -- the explicit-name form fails-safe.
    Q4_KIntegerCoreContext cx;
    cx.opName = opName;
    cx.role = role;
    cx.sizeType = sizeType;
    cx.i32ImmType = i32ImmType;
    cx.u32Type = u32Type;
    cx.i8ElemType = i8ElemType;
    cx.u8m2Type = u8m2Type;
    cx.i8m2Type = i8m2Type;
    cx.i8mf2Type = i8mf2Type;
    cx.i16m1Type = i16m1Type;
    cx.i32m2Type = i32m2Type;
    cx.i8PtrType = i8PtrType;
    cx.u8PtrType = u8PtrType;
    cx.constU32Type = constU32Type;
    cx.u32PtrType = u32PtrType;
    cx.weightPtrType = weightPtrType;
    cx.activationPtrType = weightPtrType;
    cx.subBlock = subBlock;
    cx.scalesOffset = scalesOffset;
    cx.qsOffset = 0;
    cx.q8Offset = 0;
    cx.numSubBlocks = numSubBlocks;
    cx.quarter = 8;

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // uint32_t utmp[4];  (function-scoped scratch the Region-B bit-dance fills;
    // the op DECLARES it so the brick is self-contained, vs the monolithic core
    // which declares it once above the super-block loop and shares it with the
    // dot. The 16 bytes utmp[0..3] are exactly [scales[0..7], mins[0..7]].)
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // The weight base operand IS this op's super-block pointer (single super-block
    // -- no nb = n/256 loop, no ib*144 address arithmetic). Region B's emit is the
    // SAME node sequence the monolithic core emits inside its loop; returns
    // scalesU8 = (const uint8_t *)&utmp[0].
    mlir::Value scalesU8 =
        emitQ4_KScaleMinBitDanceCore(rewriter, loc, cx, weightBase, utmpArray);

    // The store_scale_min observable: vse8 the 16 decoded [scales[0..7],
    // mins[0..7]] bytes through a function-scoped uint8_t scale_min[16] sink so
    // the lit has a clean output to CHECK. The SAME node sequence as the
    // monolithic K4a store_scale_min (vsetvl_e8m1(16) / vle8_v_u8m1 / vse8_v_u8m1
    // with the ptr = base + ib*16 arithmetic); here the sink is a local array and
    // ib is the single super-block index 0 (the address operands differ by
    // construction -- the bit-dance core above is byte-identical to the monolith).
    mlir::Type u8ElemType = emitc::OpaqueType::get(ctx, "uint8_t");
    mlir::Type scaleMinPtrType = emitc::PointerType::get(u8ElemType);
    mlir::Type scaleMinArrayType = emitc::ArrayType::get({16}, u8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("scale_min", opName, role));
    auto scaleMinVar = rewriter.create<emitc::VariableOp>(
        loc, scaleMinArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto scaleMinArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(scaleMinVar.getResult());
    mlir::Value scaleMinIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value scaleMinElem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, scaleMinArray,
                                        mlir::ValueRange{scaleMinIndex0})
            .getResult();
    mlir::Value scaleMinOutput =
        rewriter.create<emitc::ApplyOp>(loc, scaleMinPtrType, "&", scaleMinElem0)
            .getResult();

    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_scale_min"));
    std::string smSetvl = "__riscv_vsetvl_e8m1";
    mlir::Value vlsm = emitOpaqueCallBuilt(
        rewriter, loc, sizeType, smSetvl, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {sizeLit(16)};
        });
    std::string smLoad = "__riscv_vle8_v_u8m1";
    mlir::Type u8m1Type = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Value smVec =
        emitOpaqueCall(rewriter, loc, u8m1Type, smLoad,
                       mlir::ValueRange{scalesU8, vlsm}, opName, role);
    mlir::Value ib = sizeLit(0);
    mlir::Value smOff =
        rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(16));
    mlir::Value smOutPtr = rewriter.create<emitc::AddOp>(
        loc, scaleMinPtrType, scaleMinOutput, smOff);
    std::string smStore = "__riscv_vse8_v_u8m1";
    emitOpaqueCallVoid(rewriter, loc, smStore,
                       mlir::ValueRange{smOutPtr, smVec, vlsm}, opName, role);

    // The op's i32 m1 result token: the bit-dance writes utmp/scale_min as a side
    // effect, so the token has no live use; bind it to a zero literal to keep the
    // value map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[bitDance.getResult()] = resultToken;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KScaledDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    (void)avlArg; // Region C dots ONE super-block; the strips use literal vl ==
                  // stripWidth and the store uses literal 8 -- no n use.
    weftrvv::Q4KScaledDotOp scaledDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto sd = llvm::dyn_cast<weftrvv::Q4KScaledDotOp>(op))
        scaledDot = sd;
    }
    if (!scaledDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K scaled-dot body missing the op");

    mlir::Value aux8Base = valueMap.lookup(scaledDot.getAux8Base());
    mlir::Value scalesBase = valueMap.lookup(scaledDot.getScalesBase());
    mlir::Value q8Base = valueMap.lookup(scaledDot.getQ8Base());
    if (!aux8Base || !scalesBase || !q8Base)
      return rewriter.notifyMatchFailure(scaledDot,
                                         "q4_K scaled-dot ABI operand unmapped");

    llvm::StringRef opName = scaledDot.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = scaledDot.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type activationPtrType = q8Base.getType();

    // The Region-C structural facts come straight off the typed attrs (I4).
    int64_t qk = scaledDot.getQk();                  // 256
    int64_t subBlock = scaledDot.getSubBlock();      // 32
    int64_t numSubBlocks = qk / subBlock;            //  8

    // The Region-C integer-MAC LMUL anchor, sourced from the optional
    // integer_core_lmul (default "mf2" == today's byte-identical emit). l8/l16/l32
    // are the three rungs of the i8 -> i16 -> i32 widening chain; the default
    // (mf2 -> m1 -> m2) reproduces the legacy callee/type strings exactly. This
    // is the SAME single-source detail::deriveWideningChain the monolithic q4_K
    // core uses, so the auto-constructed Region C is byte-identical at every legal
    // anchor (the wide m1/m2 forms reach the q4_K capability flip: the fold-back).
    llvm::StringRef coreLmul = scaledDot.getIntegerCoreLmul().value_or("mf2");
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef l8 = wideningChain.l8;
    llvm::StringRef l16 = wideningChain.l16;
    llvm::StringRef l32 = wideningChain.l32;
    int64_t stripWidth = wideningChain.stripWidth; // 8 (mf2), 16 (m1), 32 (m2)
    int64_t numStrips = subBlock / stripWidth;     // 4 / 2 / 1
    int64_t foldGroups = wideningChain.foldGroups; // 1 (mf2), 2 (m1), 4 (m2)

    // The integer-core vector types. The Region-A unpack types (u8m2/i8m2) are
    // fixed at m2 (unused by Region C but completing the context); the Region-C
    // MAC types are derived from the l8/l16/l32 chain (i8mf2Type = vint8<l8>,
    // i16m1Type = vint16<l16>, i32m2Type = vint32<l32> -- the WIDE running aux32).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type i16m1Type =
        emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type =
        emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
    // The CANONICAL 8-lane integer accumulator type the wide aux32 folds back into
    // -- ALWAYS vint32m2_t (8 lanes at SEW32, the byte-exact Region-F contract).
    // At mf2 it equals i32m2Type (no widen).
    mlir::Type i32Canon8Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    mlir::Type u32PtrType = emitc::PointerType::get(constU32Type);

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    // The integer-core context: the SAME shape the monolithic q4_K core builds,
    // with the Region-C LMUL chain plumbed in. DESIGNATED (explicit field
    // assignment, C++17) so a future mid-struct field insertion cannot silently
    // misfeed the l8/l16/l32/stripWidth fields emitQ4_KScaledDotIntoAux32 reads.
    // q8Offset is 0: the q8 operand is the ABI activation data pointer directly
    // (single super-block; NO ib*292 stride, NO +4 quant offset -- the helper's
    // byteOffsetPtr then emits only the (const int8_t *) cast, the declared
    // ABI-input-ptr-vs-strided-base difference from the monolith's (yb + 4)).
    Q4_KIntegerCoreContext cx;
    cx.opName = opName;
    cx.role = role;
    cx.sizeType = sizeType;
    cx.i32ImmType = i32ImmType;
    cx.u32Type = u32Type;
    cx.i8ElemType = i8ElemType;
    cx.u8m2Type = u8m2Type;
    cx.i8m2Type = i8m2Type;
    cx.i8mf2Type = i8mf2Type;
    cx.i16m1Type = i16m1Type;
    cx.i32m2Type = i32m2Type;
    cx.i8PtrType = i8PtrType;
    cx.u8PtrType = u8PtrType;
    cx.constU32Type = constU32Type;
    cx.u32PtrType = u32PtrType;
    cx.weightPtrType = activationPtrType;
    cx.activationPtrType = activationPtrType;
    cx.subBlock = subBlock;
    cx.scalesOffset = 0;
    cx.qsOffset = 0;
    cx.q8Offset = 0;
    cx.numSubBlocks = numSubBlocks;
    cx.quarter = 8;
    cx.coreLmul = coreLmul;
    cx.l8 = l8;
    cx.l16 = l16;
    cx.l32 = l32;
    cx.stripWidth = stripWidth;
    cx.numStrips = numStrips;
    cx.foldGroups = foldGroups;
    cx.i32Canon8Type = i32Canon8Type;

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // int32_t aux32_out[8];  (function-scoped sink the canonical-8 aux32 is stored
    // into as the observable, vs the monolithic K4a which vse32-stores the aux32
    // through an ABI output pointer at base + ib*8. Here the sink is a local array
    // and ib is the single super-block index 0.)
    mlir::Type i32ElemType = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type aux32OutPtrType = emitc::PointerType::get(i32ElemType);
    mlir::Type aux32OutArrayType = emitc::ArrayType::get({8}, i32ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux32_out", opName, role));
    auto aux32OutVar = rewriter.create<emitc::VariableOp>(
        loc, aux32OutArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux32OutArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux32OutVar.getResult());
    mlir::Value aux32OutIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux32OutElem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux32OutArray,
                                        mlir::ValueRange{aux32OutIndex0})
            .getResult();
    mlir::Value aux32Output =
        rewriter
            .create<emitc::ApplyOp>(loc, aux32OutPtrType, "&", aux32OutElem0)
            .getResult();

    // The aux8 / scales / q8 operands ARE the already-resolved super-block data
    // pointers (single super-block; NO nb = n/256 loop). Region C's emit is the
    // SAME node sequence the monolithic core emits inside its loop; it seeds +
    // accumulates the running aux32 via the integer_core_lmul MAC chain, then
    // (at the wide m1/m2 anchors) folds it back to the canonical 8-lane vint32m2,
    // and returns that canonical-8 aux32 lvalue.
    mlir::TypedValue<emitc::LValueType> resultAux32Var =
        emitQ4_KScaledDotIntoAux32(rewriter, loc, cx, q8Base, aux8Base,
                                   scalesBase);

    // vse32_v_i32m2(aux32_out, aux32, 8);  -- store the canonical-8 aux32[8] state
    // (the SAME vse32 the monolithic K4a emits; the canonical type is vint32m2_t
    // at EVERY anchor). Void interleave (inline VL=8 literal): split the mangler
    // at the first underscore after "__riscv_" and rejoin via emitVCallVoidBuilt
    // -- byte-exact identity for "__riscv_vse32_v_i32m2".
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_aux32"));
    mlir::Value aux32Final =
        rewriter.create<emitc::LoadOp>(loc, i32Canon8Type, resultAux32Var)
            .getResult();
    emitVCallVoidBuilt(rewriter, loc, "vse32", "v_i32m2", opName, role,
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         return {aux32Output, aux32Final, sizeLit(8)};
                       });

    // The op's i32 m1 result token: the dot writes aux32_out as a side effect, so
    // the token has no live use; bind it to a zero literal to keep the map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[scaledDot.getResult()] = resultToken;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KMinTerm(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    (void)avlArg; // The MIN term is a SCALAR reduction over a SINGLE super-block
                  // (literal bsums/min counts); NO nb = n/256 loop, no n use.
    weftrvv::Q4KMinTermOp minTerm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto mt = llvm::dyn_cast<weftrvv::Q4KMinTermOp>(op))
        minTerm = mt;
    }
    if (!minTerm)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K min-term body missing the op");

    mlir::Value weightBase = valueMap.lookup(minTerm.getWeightBase());
    mlir::Value scalesBase = valueMap.lookup(minTerm.getScalesBase());
    mlir::Value activationBase = valueMap.lookup(minTerm.getActivationBase());
    if (!weightBase || !scalesBase || !activationBase)
      return rewriter.notifyMatchFailure(minTerm,
                                         "q4_K min-term ABI operand unmapped");

    llvm::StringRef opName = minTerm.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = minTerm.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constI16Type = emitc::OpaqueType::get(ctx, "const int16_t");
    mlir::Type constI16PtrType = emitc::PointerType::get(constI16Type);
    mlir::Type i32TokenType = emitc::OpaqueType::get(ctx, "int32_t");
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";
    mlir::Type activationPtrType = activationBase.getType();

    // The MIN-term structural facts come straight off the typed attrs (I4). The
    // activation d (dy) sits at byte offset 0 (q8_K) -- it is the op's own setup,
    // NOT a typed fact (always 0 for this family).
    int64_t qk = minTerm.getQk();                 // 256
    int64_t bsumsOffset = minTerm.getBsumsByteOffset();          // 260
    int64_t weightDminOffset = minTerm.getWeightDminByteOffset();//   2
    int64_t numBsums = qk / 16;                   //  16
    int64_t activationDOffset = 0;                //   0

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    // The MIN-term context: the SAME facts/types the monolithic q4_K block dot
    // builds, plumbed into the two shared MIN-term helpers (byte-identity by
    // construction).
    Q4_KMinTermContext mx;
    mx.opName = opName;
    mx.role = role;
    mx.sizeType = sizeType;
    mx.floatType = floatType;
    mx.i32Type = intType;
    mx.constU8Type = constU8Type;
    mx.constI16Type = constI16Type;
    mx.constI16PtrType = constI16PtrType;
    mx.fp16ReadCallee = fp16ReadCallee;
    mx.bsumsOffset = bsumsOffset;
    mx.weightDminOffset = weightDminOffset;
    mx.numBsums = numBsums;

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  -- the op's OWN scalar accumulator the MIN subtraction
    // lands in (vs the monolith's loop-carried sumf; here a single super-block).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // float sumf_out[1];  -- the function-scoped sink the resulting sumf is
    // stored into as the observable (vs the monolith's `*s = sumf` ABI output;
    // here a local array so the lit has output to CHECK).
    mlir::Type sumfOutArrayType = emitc::ArrayType::get({1}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf_out", opName, role));
    auto sumfOutVar = rewriter.create<emitc::VariableOp>(
        loc, sumfOutArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sumfOutArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumfOutVar.getResult());

    // The fp32 activation scale dy = *(const float *)(vy + 0), loaded ONCE (the
    // monolith's fold_activation_d, here the op's own setup; activationDOffset is
    // 0 so no add is emitted). dy feeds the dmin = fp16(x.dmin) * dy product.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "fold_activation_d"));
    mlir::Value dyAddr = activationBase;
    if (activationDOffset != 0)
      dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType,
                                             activationBase,
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
    mlir::Value dy = rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
                         .getResult();

    // The two shared MIN-term helpers (the SAME node sequence the monolithic
    // q4_K/q5_K block dot emits, here CONTIGUOUS since there is no interleaved
    // positive fold): the int16 bsums load + the scalar integer reduction sumi =
    // sum(bsums * mins) reading the activation + decoded-scales pointers, then the
    // fp16 dmin read + the `sumf -= dmin * sumi` emitc.expression reading the
    // weight pointer + dy. Byte-identical to the monolith's MIN term.
    mlir::TypedValue<emitc::LValueType> sumiVar =
        emitQ4_KMinTermBsumsDot(rewriter, loc, mx, activationBase, scalesBase);
    emitQ4_KMinTermSubtract(
        rewriter, loc, mx, weightBase, dy, sumiVar,
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));

    // sumf_out[0] = sumf;  -- store the resulting sumf state (the observable).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sumf"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value sumfOutIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp sumfOutSubscript = rewriter.create<emitc::SubscriptOp>(
        loc, sumfOutArray, mlir::ValueRange{sumfOutIndex0});
    rewriter.create<emitc::AssignOp>(loc, sumfOutSubscript.getResult(),
                                     sumfFinal);

    // The op's i32 m1 result token: the term writes sumf_out as a side effect, so
    // the token has no live use; bind it to a zero literal to keep the map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32TokenType, "0");
    valueMap[minTerm.getResult()] = resultToken;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KSumsFoldScaleD(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    (void)avlArg; // The positive fold dots ONE super-block (the canonical-8 fp
                  // sequence uses literal vl == 8); NO nb = n/256 loop, no n use.
    weftrvv::Q4KSumsFoldScaleDOp fold;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto fd = llvm::dyn_cast<weftrvv::Q4KSumsFoldScaleDOp>(op))
        fold = fd;
    }
    if (!fold)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K sums-fold body missing the op");

    mlir::Value weightBase = valueMap.lookup(fold.getWeightBase());
    mlir::Value aux32Base = valueMap.lookup(fold.getAux32Base());
    mlir::Value activationBase = valueMap.lookup(fold.getActivationBase());
    if (!weightBase || !aux32Base || !activationBase)
      return rewriter.notifyMatchFailure(fold,
                                         "q4_K sums-fold ABI operand unmapped");

    llvm::StringRef opName = fold.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = fold.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    // The CANONICAL 8-lane integer accumulator type the BRICK 3 aux32 lives in
    // and the fp fold consumes -- ALWAYS vint32m2_t (8 lanes at SEW32, the
    // byte-exact contract).
    mlir::Type i32Canon8Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i32TokenType = emitc::OpaqueType::get(ctx, "int32_t");
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";
    mlir::Type activationPtrType = activationBase.getType();

    // The positive-fold structural facts come straight off the typed attrs (I4).
    // The activation d (dy) sits at byte offset 0 (q8_K) -- it is the op's own
    // setup, NOT a typed fact (always 0 for this family).
    int64_t weightDOffset = fold.getWeightDByteOffset(); // 0
    int64_t activationDOffset = 0;                        // 0
    int64_t numLanes = 8;                                // canonical aux32/sums lanes

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the op's OWN 8-lane
    // fp32 positive accumulator (vs the monolith's loop-carried sums); seeded once.
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

    // float sums_out[8];  -- the function-scoped sink the resulting 8-lane sums
    // vector is stored into as the observable (vs the monolith's post-loop
    // horizontal fold into *s; here a local array so the lit has output to CHECK).
    mlir::Type floatPtrType = emitc::PointerType::get(floatType);
    mlir::Type sumsOutArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums_out", opName, role));
    auto sumsOutVar = rewriter.create<emitc::VariableOp>(
        loc, sumsOutArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sumsOutArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumsOutVar.getResult());

    // vint32m2_t aux32 = __riscv_vle32_v_i32m2(aux32_base, 8);  -- materialize the
    // BRICK 3 canonical-8 integer dot result from the ABI int32 pointer into the
    // aux32 lvalue (the op's own ABI-input load, vs the monolith's brick-3 helper
    // return). The fold helper then loads this lvalue exactly as the monolith does.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux32", opName, role));
    auto aux32Var = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(i32Canon8Type),
        emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value aux32Loaded = emitVCallBuilt(
        rewriter, loc, i32Canon8Type, "vle32", "v_i32m2", opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {aux32Base, sizeLit(numLanes)};
        });
    rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Loaded);

    // The fp32 activation scale dy = *(const float *)(vy + 0), loaded ONCE (the
    // monolith's fold_activation_d, here the op's own setup; activationDOffset is
    // 0 so no add is emitted). dy feeds the d = fp16(x.d) * dy product.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "fold_activation_d"));
    mlir::Value dyAddr = activationBase;
    if (activationDOffset != 0)
      dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType,
                                             activationBase,
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
    mlir::Value dy = rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
                         .getResult();

    // The shared positive-fold helper (the SAME node sequence the monolithic
    // q4_K/q5_K block dot emits between its two MIN-term halves): d = fp16(x.d) *
    // dy, af = vfcvt(aux32), pr = vfmul.vf(af, d), sums = vfadd.vv(sums, pr).
    // Byte-identical to the monolith's fold_scale_d region.
    Q4_KSumsFoldContext sx;
    sx.opName = opName;
    sx.role = role;
    sx.sizeType = sizeType;
    sx.floatType = floatType;
    sx.f32m2Type = f32m2Type;
    sx.i32Canon8Type = i32Canon8Type;
    sx.fp16ReadCallee = fp16ReadCallee;
    sx.weightDOffset = weightDOffset;
    emitQ4_KSumsFoldScaleD(
        rewriter, loc, sx, weightBase, dy,
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(aux32Var.getResult()),
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumsVar.getResult()));

    // vse32_v_f32m2(&sums_out[0], sums, 8);  -- store the resulting 8-lane sums
    // state (the observable; the monolith instead horizontally folds sums into
    // *s). Void interleave (inline VL=8 literal): split the mangler at the first
    // underscore after "__riscv_" and rejoin via emitVCallVoidBuilt.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums"));
    mlir::Value sumsOutIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sumsOutElem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sumsOutArray,
                                        mlir::ValueRange{sumsOutIndex0})
            .getResult();
    mlir::Value sumsOutBase =
        rewriter.create<emitc::ApplyOp>(loc, floatPtrType, "&", sumsOutElem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    emitVCallVoidBuilt(rewriter, loc, "vse32", "v_f32m2", opName, role,
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         return {sumsOutBase, sumsFinal, sizeLit(numLanes)};
                       });

    // The op's i32 m1 result token: the fold writes sums_out as a side effect, so
    // the token has no live use; bind it to a zero literal to keep the map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32TokenType, "0");
    valueMap[fold.getResult()] = resultToken;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitQ4_KHorizontalFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    (void)avlArg; // The post-loop fold is a fixed-order 8-lane collapse (literal
                  // vl == 8); NO nb = n/256 loop, no n use.
    weftrvv::Q4KHorizontalFoldOp fold;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto fd = llvm::dyn_cast<weftrvv::Q4KHorizontalFoldOp>(op))
        fold = fd;
    }
    if (!fold)
      return rewriter.notifyMatchFailure(
          scope, "q4_K horizontal-fold body missing the op");

    mlir::Value sumsBase = valueMap.lookup(fold.getSumsBase());
    if (!sumsBase)
      return rewriter.notifyMatchFailure(
          fold, "q4_K horizontal-fold ABI operand unmapped");

    llvm::StringRef opName = fold.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = fold.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type i32TokenType = emitc::OpaqueType::get(ctx, "int32_t");

    // The canonical fp32 lane count comes straight off the typed attr (I4).
    int64_t numLanes = fold.getNumLanes(); // 8

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // vfloat32m2_t sums = __riscv_vle32_v_f32m2(sums_base, 8);  -- the op's OWN
    // 8-lane fp32 accumulator, materialized from the ABI const float * source (vs
    // the monolith's loop-carried sums) so the post-loop fold has observable input.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumsLoaded = emitVCallBuilt(
        rewriter, loc, f32m2Type, "vle32", "v_f32m2", opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          return {sumsBase, sizeLit(numLanes)};
        });
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsLoaded);

    // float sumf = 0.0f;  -- the op's OWN scalar accumulator (vs the monolith's
    // loop-carried sumf holding the MIN subtractions; here seeded to 0 so the
    // standalone observes the pure positive horizontal sum).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // float sums8[8];  -- the function-scoped scratch the 8-lane sums is stored
    // into (vse32) before the SEQUENTIAL horizontal sum.
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // float sumf_out[1];  -- the function-scoped sink the resulting sumf is stored
    // into as the observable (vs the monolith's `*s = sumf` ABI output; here a
    // local array so the lit has output to CHECK).
    mlir::Type sumfOutArrayType = emitc::ArrayType::get({1}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf_out", opName, role));
    auto sumfOutVar = rewriter.create<emitc::VariableOp>(
        loc, sumfOutArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sumfOutArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumfOutVar.getResult());

    // The shared post-loop horizontal-fold helper (the SAME node sequence the
    // monolithic q4_K block dot emits AFTER its super-block loop): vse32 the 8-lane
    // sums into sums8, then the SEQUENTIAL ascending `sumf += sums8[l]` (l=0..7,
    // NEVER a vfredusum). Returns the final sumf. Byte-identical to the monolith's
    // store_sums_lanes + horizontal_sum region.
    Q4_KHorizontalFoldContext hx;
    hx.opName = opName;
    hx.role = role;
    hx.sizeType = sizeType;
    hx.floatType = floatType;
    hx.f32m2Type = f32m2Type;
    hx.numLanes = numLanes;
    mlir::Value sumf = emitQ4_KHorizontalFold(
        rewriter, loc, hx, sums8Array,
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumsVar.getResult()),
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));

    // sumf_out[0] = sumf;  -- store the resulting scalar (the observable; the
    // monolith instead stores it through the ABI `*s` output pointer).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sumf"));
    mlir::Value sumfOutIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp sumfOutSubscript = rewriter.create<emitc::SubscriptOp>(
        loc, sumfOutArray, mlir::ValueRange{sumfOutIndex0});
    rewriter.create<emitc::AssignOp>(loc, sumfOutSubscript.getResult(), sumf);

    // The op's i32 m1 result token: the fold writes sumf_out as a side effect, so
    // the token has no live use; bind it to a zero literal to keep the map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32TokenType, "0");
    valueMap[fold.getResult()] = resultToken;
    return mlir::success();
  }

// M-FLAT q4_K/q5_K super-block loop-scaffold emitter (milestone-2/3, W5-W7).
// Lowers the region-carrying weft_rvv.typed_super_block_block_dot_loop_body to
// the byte-exact skeleton the retired q4_K AND q5_K monoliths emitted (this typed
// emitter is now the SOLE super-block lowering for BOTH: the qh 5th-bit inject the
// retired emitQ5_KQ8_KBlockDot did is driven here by BRICK 1's optional
// weight_qh_byte_offset -> cx.hasQh, the shared CORE helpers emitting the identical
// sequence for both formats), but sourcing the in-loop
// per-super-block ADDRESSES from the region bricks' OPERANDS (anti-bypass W4)
// while REUSING the 8 shared CORE helpers so the emitted C stays byte-identical
// to the monolith by construction (same helpers, same facts, same order). This
// is the byte-exact DUAL-accumulator emit the scaffold (milestone-1) deferred:
// the honest target IR now lowers to a real emitc.func through the
// weft.exec.variant legalization (it no longer stops at the W5-W7 stub).
//
// W5 (addressing): each per-super-block base = base + ib*stride is built from a
//   brick's (base operand, block_index induction operand) via a shared per-body
//   memo (the flat blockBaseFor pattern) -- NOT the loop-invariant super-block-0.
// W6 (accumulators): the carried `sums` vfloat32m2 + `sumf` float emitc.variable
//   DUAL chain is declared + seeded ONCE outside the loop (emitc.for has no
//   iter_args); the post-loop vector->scalar collapse REUSES the BRICK 7
//   horizontal-fold helper.
// W7 (chain): the driver orchestrates the 8 CORE helpers in super-block order,
//   each fed OP-BY-OP from its region brick's operands.
mlir::LogicalResult VariantToEmitCFunc::emitTypedSuperBlockBlockDotLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto lb = llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(op))
        loopBody = lb;
    }
    if (!loopBody)
      return rewriter.notifyMatchFailure(
          scope, "q4_K super-block loop body missing the op");

    // fold_model KEYS the accumulator arity (mirrors the loop op verifier). The
    // q6_K no-min "scales_times_sumi" path carries a SINGLE `sums` vector
    // accumulator (the aux32 integer core + the no-min positive fold + a single
    // yield); dispatch it to its own emitter. The q4_K/q5_K DUAL
    // "super_block_two_level_scale_min" path (below) is UNTOUCHED (zero regression).
    if (loopBody.getFoldModel() == "scales_times_sumi")
      return emitTypedSuperBlockScalesTimesSumiLoopBody(
          rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);

    // The q2_K "scalar_scale_min" path carries a SINGLE `sumf` SCALAR accumulator
    // (the q2_K integer core producing the two scalar states isum + summs + the
    // scalar fold + a single scalar yield -- NO 8-lane sums vector, NO deferred
    // vector fold, NO post-loop horizontal add); dispatch it to its own emitter.
    // The q4_K/q5_K DUAL and q6_K SINGLE-vector paths are UNTOUCHED (zero
    // regression).
    if (loopBody.getFoldModel() == "scalar_scale_min")
      return emitTypedSuperBlockScalarScaleMinLoopBody(
          rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);

    // The iq1_s "scalar_delta_grid" path carries a SINGLE `sumf` SCALAR accumulator
    // arity-identical to q2_K's "scalar_scale_min", but the integer core is a
    // TERNARY-grid GATHER (decode_model=lookup) and the fold arithmetic is the
    // iq1_s delta fold `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)`; dispatch
    // it to its own grid emitter. The q4_K/q5_K DUAL, q6_K SINGLE-vector, and q2_K
    // scalar paths are UNTOUCHED (zero regression).
    if (loopBody.getFoldModel() == "scalar_delta_grid")
      return emitTypedSuperBlockScalarDeltaGridLoopBody(
          rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);

    // ---- Region walk (identify, no emit): the 5 in-loop bricks + the yield. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    weftrvv::Q4KNibbleUnpackOp b1;
    weftrvv::Q4KScaleMinBitDanceOp b2;
    weftrvv::Q4KScaledDotOp b3;
    weftrvv::Q4KMinTermOp b4;
    weftrvv::Q4KSumsFoldScaleDOp b6;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::Q4KNibbleUnpackOp>(bodyOp))
        b1 = o;
      else if (auto o = llvm::dyn_cast<weftrvv::Q4KScaleMinBitDanceOp>(bodyOp))
        b2 = o;
      else if (auto o = llvm::dyn_cast<weftrvv::Q4KScaledDotOp>(bodyOp))
        b3 = o;
      else if (auto o = llvm::dyn_cast<weftrvv::Q4KMinTermOp>(bodyOp))
        b4 = o;
      else if (auto o = llvm::dyn_cast<weftrvv::Q4KSumsFoldScaleDOp>(bodyOp))
        b6 = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact DUAL-accumulator
    // super-block body requires ALL 5 bricks + the dual yield, the (index, sums
    // vector, sumf scalar) entry-arg triple, and every brick's block_index tied
    // to the loop induction variable (region arg 0). This is the anti-bypass tie:
    // the emit provably tracks the region content (the block_index operand + the
    // brick base operands), not merely the loop op attrs. ----
    if (!b1 || !b2 || !b3 || !b4 || !b6 || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q4_K super-block body requires the 5 in-loop bricks "
                    "(nibble_unpack, scale_min_bit_dance, scaled_dot, min_term, "
                    "sums_fold_scale_d) and the dual yield");
    if (coreBlock.getNumArguments() != 3)
      return rewriter.notifyMatchFailure(
          loopBody, "q4_K super-block body region must carry exactly the "
                    "(super_block_index, sums, sumf) triple");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumsArg = coreBlock.getArgument(1);
    mlir::Value sumfArg = coreBlock.getArgument(2);
    if (yieldOp.getSumsNext() != sumsArg || yieldOp.getSumfNext() != sumfArg)
      return rewriter.notifyMatchFailure(
          yieldOp, "q4_K super-block yield must carry the loop-carried sums "
                   "vector and sumf scalar accumulators");
    // Every addressing brick must key its per-super-block offset off the loop
    // induction variable (region arg 0). A brick whose block_index is absent or
    // is some other value would address the loop-invariant super-block-0 (ib>0
    // all wrong), so it is fail-closed rejected -- the operand-driven gate.
    std::array<mlir::Value, 5> bricksBlockIndex{
        b1.getBlockIndex(), b2.getBlockIndex(), b3.getBlockIndex(),
        b4.getBlockIndex(), b6.getBlockIndex()};
    for (mlir::Value bi : bricksBlockIndex)
      if (bi != sbIndex)
        return rewriter.notifyMatchFailure(
            loopBody, "every q4_K super-block brick's block_index must be the "
                      "loop induction variable (region arg 0) so the emit "
                      "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same four the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody, "q4_K super-block loop ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP
    // OP (the byte-exact schedule shape knobs, exactly like the flat loop body
    // reads deriveBlockDotFacts off the loop op); the per-region byte offsets +
    // the integer-core LMUL come off the BRICKS that own them (I4 mirror), so the
    // facts are sourced from the same typed surface the standalone bricks read --
    // byte-identity by construction. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = b1.getSubBlock();                        // 32
    int64_t weightStride = loopBody.getWeightBlockStride();     // 144
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = b6.getWeightDByteOffset();          //   0
    int64_t weightDminOffset = b4.getWeightDminByteOffset();    //   2
    int64_t scalesOffset = b2.getWeightScalesByteOffset();      //   4
    int64_t qsOffset = b1.getWeightQsByteOffset();              //  16 q4/48 q5
    // W3 (q5_K flip): the ONLY q5_K-vs-q4_K difference is the qh 5th-bit plane.
    // BRICK 1 carries the OPTIONAL weight_qh_byte_offset -- PRESENT selects q5_K
    // (inject the 5th bit @ this offset), ABSENT keeps q4_K (byte-identical, no qh
    // load, no inject). Pure emitter-side wiring: the shared leaf helper
    // emitQ4_KPlainNibbleUnpack already emits the byte-exact qh inject under
    // cx.hasQh (the SAME node sequence the retired q5_K monolith emitted), so this
    // flip only threads the brick attr into the core context.
    bool hasQh = b1.getWeightQhByteOffset().has_value();
    int64_t qhOffset =
        static_cast<int64_t>(b1.getWeightQhByteOffset().value_or(0)); // 16 (q5_K)
    int64_t activationDOffset = 0;                              //   0
    int64_t q8Offset =
        static_cast<int64_t>(b3.getActivationQuantByteOffset().value_or(4)); // 4
    int64_t bsumsOffset = b4.getBsumsByteOffset();              // 260
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t quarter = 8;                                        // 8-elem quarters
    int64_t numLanes = 8;                                       // aux32/sums lanes
    int64_t numBsums = qk / 16;                                 //  16

    // The Region-C integer-MAC LMUL anchor, sourced from the BRICK 3 scaled dot
    // (the brick that owns the integer_core_lmul knob), default "mf2". Same
    // detail::deriveWideningChain the monolith uses -> byte-identical at every
    // legal anchor.
    llvm::StringRef coreLmul = b3.getIntegerCoreLmul().value_or("mf2");
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef l8 = wideningChain.l8;
    llvm::StringRef l16 = wideningChain.l16;
    llvm::StringRef l32 = wideningChain.l32;
    int64_t stripWidth = wideningChain.stripWidth;
    int64_t numStrips = subBlock / stripWidth;
    int64_t foldGroups = wideningChain.foldGroups;

    // The integer-core + fp fold types (copied VERBATIM from the monolith so the
    // emitted C references the SAME Type instances).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, ("vint8" + l8 + "_t").str());
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, ("vint16" + l16 + "_t").str());
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, ("vint32" + l32 + "_t").str());
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

    // int8_t aux8[256];  (function-scoped scratch, shared with the integer core;
    // W7 threads it as function-level declaration, exactly like the monolith.)
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

    // uint32_t utmp[4];
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // float sums8[8];
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- W6 carried VECTOR
    // accumulator, declared + zeroed ONCE OUTSIDE the super-block loop (the typed
    // region arg `sums` lowers to this emitc.variable lvalue; emitc.for has no
    // iter_args, so the helpers mutate it in place across iterations).
    (void)sumsArg;
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

    // float sumf = 0.0f;  -- W6 carried SCALAR accumulator, declared + zeroed ONCE
    // OUTSIDE the loop (the typed region arg `sumf` lowers to this lvalue).
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // The integer-core context (ONE cx, the SAME shape + facts the monolith
    // builds) -> the 3 shared leaf helpers emit byte-identical Region A/B/C C.
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        activationPtrType, subBlock,  scalesOffset,  qsOffset,
        q8Offset,      numSubBlocks,  quarter,       coreLmul,
        l8,            l16,           l32,           stripWidth,
        numStrips,     foldGroups,    i32Canon8Type, hasQh,
        qhOffset};

    // The BRICK 4 MIN-term + BRICK 6 positive-fold contexts (the SAME facts/types
    // the monolith builds; the shared helpers are byte-identical by construction).
    Q4_KMinTermContext mx;
    mx.opName = opName;
    mx.role = role;
    mx.sizeType = sizeType;
    mx.floatType = floatType;
    mx.i32Type = i32Type;
    mx.constU8Type = constU8Type;
    mx.constI16Type = constI16Type;
    mx.constI16PtrType = constI16PtrType;
    mx.fp16ReadCallee = fp16ReadCallee;
    mx.bsumsOffset = bsumsOffset;
    mx.weightDminOffset = weightDminOffset;
    mx.numBsums = numBsums;

    Q4_KSumsFoldContext sx;
    sx.opName = opName;
    sx.role = role;
    sx.sizeType = sizeType;
    sx.floatType = floatType;
    sx.f32m2Type = f32m2Type;
    sx.i32Canon8Type = i32Canon8Type;
    sx.fp16ReadCallee = fp16ReadCallee;
    sx.weightDOffset = weightDOffset;

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W5: per-super-block base = base + ib*stride, built OP-BY-OP from each
      // brick's (base operand, block_index operand) through a per-iteration memo
      // (the flat blockBaseFor pattern). Correct wiring (all weight bricks -> the
      // ABI weight base, all activation bricks -> the ABI activation base, all
      // block_index == the IV) collapses to exactly TWO emitted bases -- xb
      // (super_block_base_x) and yb (super_block_base_y) -- byte-identical to the
      // monolith's blockBaseValue. A CHANGED brick base operand keys a DIFFERENT
      // memo entry and emits a DIFFERENT base (anti-bypass: the emit tracks the
      // operand, not a gate-only check).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact
      // order (weight x before activation y), each from the brick that owns that
      // base in the monolith: xb from BRICK 1 (nibble unpack weight base), yb from
      // BRICK 3 (scaled dot q8 base).
      mlir::Value xb = blockBaseFor(b1.getWeightBase(), b1.getBlockIndex(),
                                    weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseFor(b3.getQ8Base(), b3.getBlockIndex(),
                                    activationStride, "super_block_base_y");

      // ---- Region A/B/C: the 3 shared leaf helpers, in the monolith's order,
      // each fed its per-brick base (BRICK 1 -> unpack, BRICK 2 -> bit-dance,
      // BRICK 3 -> scaled dot). Aux32Core is exactly this sequence with a null
      // scaleMinOutput; inlining the 3 leaves makes bricks 1/2/3 independently
      // operand-driven while staying byte-identical. ----
      emitQ4_KPlainNibbleUnpack(rewriter, loc, cx, xb, aux8Array);
      mlir::Value scalesU8 = emitQ4_KScaleMinBitDanceCore(
          rewriter, loc, cx,
          blockBaseFor(b2.getWeightBase(), b2.getBlockIndex(), weightStride,
                       "super_block_base_x"),
          utmpArray);
      mlir::TypedValue<emitc::LValueType> aux32Var = emitQ4_KScaledDotIntoAux32(
          rewriter, loc, cx,
          blockBaseFor(b3.getQ8Base(), b3.getBlockIndex(), activationStride,
                       "super_block_base_y"),
          aux8Base, scalesU8);

      // ---- fp32 activation scale dy = *(const float *)(yb + 0), loaded ONCE. ----
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

      // ---- BRICK 4 MIN term, FIRST half: the int16 bsums reduction (yb driven
      // by BRICK 4's activation base). ----
      mlir::TypedValue<emitc::LValueType> sumiVar = emitQ4_KMinTermBsumsDot(
          rewriter, loc, mx,
          blockBaseFor(b4.getActivationBase(), b4.getBlockIndex(),
                       activationStride, "super_block_base_y"),
          scalesU8);

      // ---- BRICK 6 deferred positive fold (xb driven by BRICK 6's weight base):
      // sums += (fp16(x.d) * dy) * (float)aux32. Slotted BETWEEN the two MIN-term
      // halves exactly as the monolith, data-independent of the MIN chain. ----
      emitQ4_KSumsFoldScaleD(
          rewriter, loc, sx,
          blockBaseFor(b6.getWeightBase(), b6.getBlockIndex(), weightStride,
                       "super_block_base_x"),
          dy, aux32Var,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumsVar.getResult()));

      // ---- BRICK 4 MIN term, SECOND half: sumf -= dmin * (float)sumi (xb driven
      // by BRICK 4's weight base). ----
      emitQ4_KMinTermSubtract(
          rewriter, loc, mx,
          blockBaseFor(b4.getWeightBase(), b4.getBlockIndex(), weightStride,
                       "super_block_base_x"),
          dy, sumiVar,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // ---- W6 post-loop vector->scalar collapse: REUSE the BRICK 7 horizontal-fold
    // helper (moved OUT of the region to post-loop), byte-identical to the
    // monolith's post-loop fold. ----
    Q4_KHorizontalFoldContext hx;
    hx.opName = opName;
    hx.role = role;
    hx.sizeType = sizeType;
    hx.floatType = floatType;
    hx.f32m2Type = f32m2Type;
    hx.numLanes = numLanes;
    mlir::Value sumf = emitQ4_KHorizontalFold(
        rewriter, loc, hx, sums8Array,
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumsVar.getResult()),
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody, "q4_K super-block loop output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);
    return mlir::success();
  }

// M-FLAT q6_K super-block SINGLE-accumulator loop emitter (milestone-2, W-D).
// Lowers the region-carrying weft_rvv.typed_super_block_block_dot_loop_body whose
// fold_model is "scales_times_sumi" (the q6_K no-min path) to the byte-exact
// skeleton the retired q6_K monolith emitQ6_KQ8_KBlockDot emitted: the SAME q6_K
// integer core (emitQ6_KSuperBlockAux32Core -- the 2-bit qh + 8-bit signed scale
// unpack into aux32), the SAME no-min positive fold (sums += fp16(x.d)*y.d*
// (float)aux32), the SAME post-loop SEQUENTIAL horizontal add into *s -- with a
// SINGLE `sums` VECTOR accumulator and NO scalar MIN chain. Unlike the q4_K/q5_K
// dual path, there is NO sumf scalar, NO bit-dance/min-term bricks: the region
// carries exactly the aux32 integer-core brick + the reused positive-fold brick +
// the single yield. The in-loop per-super-block ADDRESSES come from the region
// bricks' OPERANDS (anti-bypass W-E) while the emitted C stays byte-identical to
// the monolith by construction (same core helper, same facts, same order). This
// is the SOLE q6_K super-block lowering.
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalesTimesSumiLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the aux32 core + fold brick + yield.
    // The single-vector no-min body is shared by TWO formats whose ONLY difference
    // is the integer-core brick: q6_K (q6_k_q8_k_aux32_partial -- 2-bit qh + direct
    // int8 scale) and q3_K (q3_k_q8_k_aux32_partial -- 2-bit + subtractive-hmask
    // decode + SIGNED 6-bit scale). Both reuse the SAME no-min positive fold
    // (q4_k_sums_fold_scale_d) + single `sums` yield.
    weftrvv::GgmlBlockDotQ6KQ8KAux32Op aux32Op;
    weftrvv::GgmlBlockDotQ3KQ8KAux32Op q3Aux32Op;
    weftrvv::Q4KSumsFoldScaleDOp foldOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::GgmlBlockDotQ6KQ8KAux32Op>(bodyOp))
        aux32Op = o;
      else if (auto o = llvm::dyn_cast<weftrvv::GgmlBlockDotQ3KQ8KAux32Op>(bodyOp))
        q3Aux32Op = o;
      else if (auto o = llvm::dyn_cast<weftrvv::Q4KSumsFoldScaleDOp>(bodyOp))
        foldOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });
    const bool isQ3K = static_cast<bool>(q3Aux32Op);

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SINGLE-accumulator
    // no-min super-block body requires EITHER format's aux32 integer-core brick +
    // the reused positive-fold brick + the SINGLE yield, the (index, sums vector)
    // entry-arg pair, and every brick's block_index tied to the loop induction
    // variable (region arg 0) -- the anti-bypass tie (the emit provably tracks the
    // region content, not merely the loop op attrs). ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if ((!aux32Op && !q3Aux32Op) || (aux32Op && q3Aux32Op) || !foldOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "no-min super-block single-accumulator body requires EXACTLY "
                    "one aux32 integer core (q6_K or q3_K) + the reused "
                    "positive-fold brick + the single yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "no-min super-block single-accumulator body region must carry "
                    "exactly the (super_block_index, sums) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumsArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumsArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "no-min super-block single yield must carry the loop-carried "
                   "sums vector ALONE (no sumf scalar under the no-min fold)");
    // The aux32 brick's base/block_index operands, abstracted over the two formats.
    mlir::Value aux32WeightBase =
        isQ3K ? q3Aux32Op.getWeightBase() : aux32Op.getWeightBase();
    mlir::Value aux32ActivationBase =
        isQ3K ? q3Aux32Op.getActivationBase() : aux32Op.getActivationBase();
    mlir::Value aux32BlockIndex =
        isQ3K ? q3Aux32Op.getBlockIndex() : aux32Op.getBlockIndex();
    std::array<mlir::Value, 2> bricksBlockIndex{aux32BlockIndex,
                                                foldOp.getBlockIndex()};
    for (mlir::Value bi : bricksBlockIndex)
      if (bi != sbIndex)
        return rewriter.notifyMatchFailure(
            loopBody, "every no-min super-block brick's block_index must be the "
                      "loop induction variable (region arg 0) so the emit "
                      "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same four the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody, "q6_K super-block single-accumulator ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP
    // OP (the byte-exact schedule shape knobs, exactly like the dual body); the
    // per-region byte offsets come off the BRICK that owns them (I4 mirror) -- the
    // aux32 op owns the qh(q6_K) OR hmask/qs(q3_K) + scales/q8 offsets + sub-block
    // shape, the fold brick owns the fp16 weight d offset (208 q6_K / 108 q3_K) --
    // so the facts are sourced from the same typed surface the standalone bricks
    // read (byte-identity by construction). ----
    int64_t qk = loopBody.getQk();                          // 256
    int64_t subBlock =
        isQ3K ? q3Aux32Op.getSubBlock() : aux32Op.getSubBlock();     //  16
    int64_t weightStride = loopBody.getWeightBlockStride();         // 210 / 110
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    // q6_K owns qh@128; q3_K owns hmask@0 + qs@32 (the qh field is unused for q3_K
    // and vice versa -- read whichever the present brick carries).
    int64_t qhOffset = isQ3K ? 0 : aux32Op.getWeightQhByteOffset(); // 128
    int64_t hmaskOffset = isQ3K ? q3Aux32Op.getWeightHmaskByteOffset() : 0; // 0
    int64_t qsOffset = isQ3K ? q3Aux32Op.getWeightQsByteOffset() : 0;       // 32
    int64_t scalesOffset = isQ3K ? q3Aux32Op.getWeightScalesByteOffset()
                                 : aux32Op.getWeightScalesByteOffset(); // 192/96
    int64_t q8Offset = isQ3K ? q3Aux32Op.getActivationQuantByteOffset()
                             : aux32Op.getActivationQuantByteOffset(); //   4
    int64_t weightDOffset = foldOp.getWeightDByteOffset();  // 208 / 108
    int64_t activationDOffset = 0;                          //   0
    int64_t numSubBlocks = qk / subBlock;                   //  16
    int64_t half = subBlock / 2;                            //   8
    int64_t numLanes = numSubBlocks / 2;                    //   8 aux32/sums lanes

    // The integer-core + fp fold types (copied VERBATIM from the monolith so the
    // emitted C references the SAME Type instances).
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

    // int8_t aux8[256];  (function-scoped scratch, shared with the integer core.)
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

    // uint32_t utmp[4];  (q3_K ONLY -- the signed 6-bit scale bit-dance output
    // words; the 16 SIGNED 6-bit scales are read as the int8 byte alias of these 4
    // words. Declared HERE -- between aux8 and sums8 -- exactly where the retired
    // q3_K monolith emitQ3_KQ8_KBlockDot placed it, so the constructed emit is
    // byte-identical. q6_K's direct int8 scale needs no such scratch.)
    mlir::TypedValue<emitc::ArrayType> utmpArray = nullptr;
    if (isQ3K) {
      mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
      mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("utmp", opName, role));
      auto utmpVar = rewriter.create<emitc::VariableOp>(
          loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
      utmpArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());
    }

    // float sums8[8];  (function-scoped scratch the 8-lane fp32 accumulator vector
    // is stored into before the SEQUENTIAL horizontal sum).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 accumulator, declared + zeroed ONCE OUTSIDE the super-block loop (the
    // typed region arg `sums` lowers to this emitc.variable lvalue; emitc.for has
    // no iter_args, so the fold mutates it in place across iterations). NO sumf
    // scalar (the no-min single-accumulator path).
    (void)sumsArg;
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

    // The format-specific integer-core context (identical unpack + sub-block-loop
    // nodes at the default mf2 -- the aux32 op carries no integer_core_lmul knob,
    // so the LITERAL mf2 chain is passed, byte-identical to the untuned monolith).
    // Exactly one is consumed below (keyed by isQ3K); the unused one is inert.
    Q6_KIntegerCoreContext cx6{
        opName,        role,          sizeType,         i32ImmType,
        i8ElemType,    u8m2Type,      i8m2Type,         i8mf2Type,
        i16m1Type,     i32m2Type,     i8PtrType,        u8PtrType,
        weightPtrType, activationPtrType, subBlock,     qhOffset,
        scalesOffset,  q8Offset,      numSubBlocks,     half,
        /*coreLmul=*/"mf2", /*l8=*/"mf2", /*l16=*/"m1", /*l32=*/"m2",
        /*i8WideType=*/i8mf2Type, /*i16WideType=*/i16m1Type,
        /*i32WideType=*/i32m2Type, /*stripWidth=*/8, /*foldGroups=*/1};
    Q3_KIntegerCoreContext cx3{opName,        role,        sizeType,
                               weightPtrType, activationPtrType, subBlock,
                               hmaskOffset,   qsOffset,    scalesOffset,
                               q8Offset,      numSubBlocks, half,
                               /*coreLmul=*/"mf2"};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built OP-BY-OP from each
      // brick's (base operand, block_index operand) through a per-iteration memo.
      // Correct wiring (both bricks -> the ABI weight/activation bases, all
      // block_index == the IV) collapses to exactly TWO emitted bases -- xb and yb
      // -- byte-identical to the monolith's blockBaseValue. A CHANGED brick base
      // operand keys a DIFFERENT memo entry and emits a DIFFERENT base (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact
      // order (weight x before activation y), from the aux32 core brick that owns
      // both bases.
      mlir::Value xb = blockBaseFor(aux32WeightBase, aux32BlockIndex,
                                    weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseFor(aux32ActivationBase, aux32BlockIndex,
                                    activationStride, "super_block_base_y");

      // The format-keyed integer core: unpack -> aux8 -> per-sub-block i32 dot,
      // returning the per-super-block aux32[8] lvalue. q3_K also stages its signed
      // 6-bit scale dance through the utmp[4] scratch.
      mlir::TypedValue<emitc::LValueType> aux32Var =
          isQ3K ? emitQ3_KSuperBlockAux32Core(rewriter, loc, cx3, xb, yb,
                                              aux8Array, aux8Base, utmpArray)
                : emitQ6_KSuperBlockAux32Core(rewriter, loc, cx6, xb, yb,
                                              aux8Array, aux8Base);

      // ---- (C) the DEFERRED fp32 positive fold (no min), byte-identical to the
      // monolith. The fold's weight/activation bases come from the FOLD BRICK's
      // operands (via the memo -> the SAME xb/yb), so the fold is operand-driven. --
      // float d = (float)*(const _Float16 *)(xb + 208) * *(const float *)(yb + 0);
      mlir::Value foldXb =
          blockBaseFor(foldOp.getWeightBase(), foldOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value foldYb =
          blockBaseFor(foldOp.getActivationBase(), foldOp.getBlockIndex(),
                       activationStride, "super_block_base_y");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = foldXb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, foldXb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      // *(const float *)(yb + 0): cast the activation base to const float * and
      // load element 0 (the fp32 q8_K scale d).
      mlir::Value dyAddr = foldYb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, foldYb,
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
      // d = dx * dy  (fp16->fp32 weight scale times fp32 activation scale).
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
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

    // ---- (D) the SEQUENTIAL horizontal sum, l = 0 .. 7 (byte-identical to the
    // monolith: the no-min single-accumulator path seeds sumf from a FRESH 0.0f
    // literal -- there is no carried sumf lvalue). ----
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
      return rewriter.notifyMatchFailure(
          loopBody, "q6_K super-block single-accumulator output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);
    return mlir::success();
  }

// M-FLAT q2_K super-block SCALAR-accumulator loop emitter (milestone-2, W-D).
// Lowers the region-carrying weft_rvv.typed_super_block_block_dot_loop_body whose
// fold_model is "scalar_scale_min" (the q2_K path) to the byte-exact skeleton the
// retired q2_K monolith emitQ2_KQ8_KBlockDot emitted: the SAME q2_K integer core
// (emitQ2_KSuperBlockIntegerCore -- the 2-bit unpack + plain uint4-nibble scale/
// min + per-sub-block scalar i32 dot into isum/summs), the SAME scalar fold
// (emitQ2_KScalarFold -- sumf += dall*isum - dmin*summs, fp16 d@80/dmin@82), the
// SAME `*s = sumf` store -- with a SINGLE `sumf` SCALAR accumulator and NO 8-lane
// `sums` vector, NO deferred vector fold, NO post-loop horizontal add. Unlike the
// q4_K/q5_K dual path and the q6_K single-vector path, the region carries exactly
// the q2_K integer-core brick + a single scalar yield: the scalar fold has no
// separate fold brick (its d@80/dmin@82 offsets are q2_K constants), so it is
// emitter-inlined. The in-loop per-super-block ADDRESSES come from the integer
// core brick's OPERANDS (anti-bypass W-E) while the emitted C stays byte-identical
// to the monolith by construction (same core/fold helpers, same facts, same
// order). This is the SOLE q2_K super-block lowering.
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarScaleMinLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the q2_K integer core + yield. ----
    weftrvv::GgmlBlockDotQ2KQ8KIntegerCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::GgmlBlockDotQ2KQ8KIntegerCoreOp>(
              bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // super-block body requires the q2_K integer-core brick + the SINGLE scalar
    // yield, the (index, sumf scalar) entry-arg pair, and the brick's block_index
    // tied to the loop induction variable (region arg 0) -- the anti-bypass tie
    // (the emit provably tracks the region content, not merely the loop op attrs).
    // ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q2_K super-block scalar-accumulator body requires the q2_K "
                    "integer-core brick + the single scalar yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "q2_K super-block scalar-accumulator body region must carry "
                    "exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "q2_K super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the q2_K super-block integer-core brick's block_index must "
                    "be the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same four the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody, "q2_K super-block scalar-accumulator ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP
    // OP (the byte-exact schedule shape knobs); the per-region byte offsets + the
    // sub-block shape come off the q2_K integer-core BRICK that owns them (I4
    // mirror). The SCALAR fold's fp16 weight d @80 / dmin @82 and fp32 activation d
    // @0 are FIXED block_q2_K constants (the scalar fold has no separate fold brick
    // in the region -- it is emitter-inlined), read here exactly as the monolith
    // read them off its own op attrs. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  16
    int64_t weightStride = loopBody.getWeightBlockStride();     //  84
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t scalesOffset = coreOp.getWeightScalesByteOffset();  //   0
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //  16
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t bsumsOffset = coreOp.getActivationBsumsByteOffset();// 260
    int64_t numSubBlocks = qk / subBlock;                       //  16
    int64_t weightDOffset = 80;                                 //  80 (fp16 x.d)
    int64_t weightDminOffset = 82;                              //  82 (fp16 x.dmin)
    int64_t activationDOffset = 0;                              //   0 (fp32 y.d)

    // The integer-core + fold types (copied VERBATIM from the monolith so the
    // emitted C references the SAME Type instances).
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

    Q2_KIntegerCoreContext cx{opName,
                              role,
                              sizeType,
                              i32Type,
                              i32ImmType,
                              floatType,
                              u8m2Type,
                              i8m2Type,
                              i8m1Type,
                              i16m2Type,
                              i32m1Type,
                              i8ElemType,
                              i8PtrType,
                              u8PtrType,
                              constU8Type,
                              constI16Type,
                              constI16PtrType,
                              weightPtrType,
                              activationPtrType,
                              fp16ReadCallee,
                              subBlock,
                              numSubBlocks,
                              qk,
                              scalesOffset,
                              qsOffset,
                              q8Offset,
                              bsumsOffset,
                              weightDOffset,
                              weightDminOffset,
                              activationDOffset};

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, shared with the integer core.)
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

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region
    // arg `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args,
    // so the fold mutates it in place across iterations). NO 8-lane `sums` vector.
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the integer core
      // brick's (base operand, block_index operand) through a per-iteration memo.
      // Correct wiring (weight base -> ABI weight, activation base -> ABI
      // activation, block_index == the IV) collapses to exactly TWO emitted bases
      // -- xb (super_block_base_x) and yb (super_block_base_y) -- byte-identical to
      // the monolith's blockBaseValue. A CHANGED brick base operand keys a
      // DIFFERENT memo entry and emits a DIFFERENT base (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact
      // order (weight x before activation y), from the integer core brick that
      // owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared q2_K integer core (2-bit unpack + plain-nibble scale/min +
      // 16x16 scalar dot) -> the two per-super-block SCALAR states (isum, summs),
      // then the shared scalar fold sumf += dall*isum - dmin*summs. Both are the
      // SAME byte-exact helpers the retired monolith called.
      auto [isumVar, summsVar] = emitQ2_KSuperBlockIntegerCore(
          rewriter, loc, cx, xb, yb, aux8Array, aux8Base);
      emitQ2_KScalarFold(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()),
          isumVar, summsVar);
    }

    // *s = sumf;  (structured scalar store through the float * output pointer).
    mlir::Value sumf =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody, "q2_K super-block scalar-accumulator output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);
    return mlir::success();
  }

// M-FLAT iq1_s super-block SCALAR-accumulator GRID emitter (the flip lowering, M3).
// Lowers the region-carrying weft_rvv.typed_super_block_block_dot_loop_body whose
// fold_model is "scalar_delta_grid" to the byte-exact skeleton the (now-retired)
// monolith emitIQ1SQ8KBlockDot emitted: the `static const uint64_t weft_iq1s_grid[2048]`
// TERNARY grid decl (keyed off the grid-core brick op identity from the canonical
// kIQ1SGrid), the `sumf` float SCALAR accumulator seeded ONCE outside the loop (NO
// 8-lane `sums` vector), nb = n / QK_K, the `weft_iq1s_grid` base literal, the outer
// emitc.for over nb, and (post-loop) the `*s` store. The in-loop per-super-block body
// is emitted by the SHARED emitIQ1SSuperBlockGridBody helper (the same one the retired
// monolith called), sourcing the per-super-block ADDRESSES from the grid-core brick's
// (base, block_index) OPERANDS via a per-body memo (anti-bypass W4), so the emitted C
// is byte-identical to the retired monolith by construction (same grid decl, same body
// helper, same facts, same order) modulo the source-op provenance token + the func
// name. M3 (the flip): the front door now constructs this typed body as the SOLE
// representation (the monolith op + emitter + verifier were retired the same action).
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // fold_model "scalar_delta_grid" covers BOTH iq1_s and its sibling iq1_m (the SAME
    // SCALAR-accumulator ternary-grid arity + scalar delta fold). Disambiguate by the
    // in-region brick op identity: an iq1_m grid-core brick routes to the iq1_m
    // emitter (packed-scale reconstruct + half-split grid dot + per-group four-sign
    // delta + the `weft_iq1m_grid` decl); otherwise this iq1_s path proceeds.
    {
      bool hasIq1mCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ1MQ8KGridCoreOp) {
        hasIq1mCore = true;
      });
      if (hasIq1mCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq1M(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // iq3_xxs (iq1_s grid sibling): an iq3_xxs GRID-of-4 core brick routes to the
    // iq3_xxs emitter (the i32 iq3xxs_grid vluxei16 gather + aux32 4-bit-scale +
    // 4-sign-group ksigns decode + the `weft_iq3xxs_grid`/`weft_iq3xxs_ksigns` decls +
    // the trailing 0.25f factor); otherwise the iq1_s path proceeds.
    {
      bool hasIq3xxsCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp) {
        hasIq3xxsCore = true;
      });
      if (hasIq3xxsCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // iq2_xxs (iq1_s grid sibling, SIGN-PLANE signs64 variant): an iq2_xxs GRID-of-8 core
    // brick routes to the iq2_xxs emitter (the i64 iq2xxs_grid vluxei16_v_i64<core> gather
    // + the SECOND signs64 vluxei16 gather over the DERIVED keven_signs_q2xs sign plane +
    // the aux1 4-bit-scale + 4-sign-group decode + the `weft_iq2xxs_grid`/`
    // weft_iq2xxs_signs64` decls + the trailing 0.125f factor); otherwise the iq1_s path
    // proceeds. Carries the Win-A m2/m1 gearbox on the brick's integer_core_lmul.
    {
      bool hasIq2xxsCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp) {
        hasIq2xxsCore = true;
      });
      if (hasIq2xxsCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // iq2_xs (iq2_xxs grid sibling, SIGN-PLANE signs64 variant, PER-HALF explicit scale):
    // an iq2_xs per-half-scale GRID core brick routes to the iq2_xs emitter (the 512-entry
    // iq2xs_grid vluxei16_v_i64m1 gather indexed by `w & 511` + the SECOND signs64 vluxei16
    // gather over the DERIVED keven_signs_q2xs sign plane keyed by `w >> 9` + the EXPLICIT
    // per-sub-block 4-bit scales[8] two-half split ls1/ls2 + the `weft_iq2xs_grid`/`
    // weft_iq2xs_signs64` decls + the trailing 0.125f factor); otherwise the iq1_s path
    // proceeds. NO gearbox -- fixed 16-lane per-half shape.
    {
      bool hasIq2xsCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ2XSQ8KGridCoreOp) {
        hasIq2xsCore = true;
      });
      if (hasIq2xsCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xs(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // iq2_s (iq2_xs grid sibling, SIGN-PLANE explicit-signs variant, PER-HALF explicit
    // scale): an iq2_s per-half-scale GRID core brick routes to the iq2_s emitter (the
    // 1024-entry iq2s_grid vluxei16_v_i64m1 gather indexed by `qs[l] | ((qh<<(8-2l))&0x300)`
    // + the SECOND signs256 vluxei16 gather over the UNIVERSAL explicit-sign-byte plane keyed
    // by the raw sign byte + the EXPLICIT per-sub-block 4-bit scales[8] two-half split
    // ls1/ls2 + the `weft_iq2s_grid`/`weft_iq2s_signs256` decls + the trailing 0.125f
    // factor); otherwise the iq1_s path proceeds. NO gearbox -- fixed 16-lane per-half shape.
    {
      bool hasIq2sCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ2SQ8KGridCoreOp) {
        hasIq2sCore = true;
      });
      if (hasIq2sCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq2s(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // iq3_s (iq3_xxs grid sibling, EXPLICIT-SIGNS variant, qh 9th-bit inject, explicit
    // two-nibble scales): an iq3_s GRID-of-4 core brick routes to the iq3_s emitter (the
    // 512-entry iq3s_grid vluxei16_v_i32m1 gather indexed by `qs[l] | ((qh<<(8-2l))&256)`
    // + the EXPLICIT per-sub-block sign bytes at offset 74 folded via the inline kmask
    // {1<<j} + the explicit two-nibble scales at offset 106 + the `weft_iq3s_grid` decl +
    // NO trailing factor); otherwise the iq1_s path proceeds. NO gearbox -- fixed grid-of-4
    // 8-lane shape, NO ksigns plane (the signs are an explicit memory region).
    {
      bool hasIq3sCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ3SQ8KGridCoreOp) {
        hasIq3sCore = true;
      });
      if (hasIq3sCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq3s(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // iq4_xs (flat iq4_nl CODEBOOK sibling, SUPER-BLOCK rung): an iq4_xs codebook-core
    // brick routes to the iq4_xs emitter (the 16-entry non-linear int8 weft_iq4_xs_kvalues
    // codebook broadcast + per-sub-block vand/vsrl nibble split + vrgather_vv_i8m1 gather +
    // asymmetric vwmul/vwmacc widening product + seed-0 vwredsum, wrapped in the q4_K-style
    // super-block SIGNED 6-bit scale bit-dance ls = ((scales_l>>...)&0xf)|(((scales_h>>...)
    // &0x3)<<4) biased -32, folded PER-SUB-BLOCK in float `sumf += (d4d8*(ls-32))*sumi`
    // with NO trailing factor); otherwise the iq1_s path proceeds. NO gearbox -- the
    // codebook gather pins m1. UNLIKE the grid siblings the fold runs per-sub-block in
    // float, but the single-scalar accumulator arity (fold_model "scalar_delta_grid") is
    // identical, and the whole body is emitter-inlined keyed off the codebook-core brick.
    {
      bool hasIq4xsCore = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotIQ4XSQ8KCodebookCoreOp) {
        hasIq4xsCore = true;
      });
      if (hasIq4xsCore)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyIq4xs(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // tq2_0 (the FIRST TQ-family member, ARITHMETIC 2-bit ternary decode): a tq2_0 FUSED
    // 2-bit ternary integer-core brick routes to the tq2_0 emitter (the 32-byte qs chunk
    // load + the 4 2-bit planes each unpacked to 32 ternary lanes via vand/vsrl + the `-1`
    // bias vsub + vwmacc DIRECTLY against the matching 32 q8 lanes into a wide i16 accumulator
    // + ONE vwredsum per chunk into the per-super-block scalar sumi, then the emitter-inlined
    // scalar fold `sumf += (float)sumi * d`, d = fp16(x.d @64) * y.d @0, NO trailing factor);
    // otherwise the iq1_s path proceeds. tq2_0 SHARES weight_block_stride 66 with iq2_xxs but
    // dispatches by this DISTINCT brick op TYPE, so there is no stride ambiguity. Carries the
    // Win-A m2/m1 gearbox on the brick's integer_core_lmul.
    {
      bool hasTq20Core = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotTQ20Q8KTernaryCoreOp) {
        hasTq20Core = true;
      });
      if (hasTq20Core)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyTQ20(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // tq1_0 (the SECOND TQ-family member, ARITHMETIC BASE-3 ternary decode): a tq1_0 BASE-3
    // ternary integer-core brick routes to the tq1_0 emitter (the qs main/tail + qh base-3
    // trit unpack -- `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` -- into an
    // element-ordered aux8[256], then the flat-256 widened i8*i8 dot -- vle8 i8 x q8 i8 ->
    // vwmul i16 -> vwredsum i32 -- into the per-super-block scalar sumi, then the
    // emitter-inlined scalar fold `sumf += (float)sumi * d`, d = fp16(x.d @52) * y.d @0, NO
    // trailing factor); otherwise the iq1_s path proceeds. tq1_0's weight_block_stride 54 is
    // UNIQUE, and this DISTINCT base-3 brick op TYPE disambiguates it. Carries the Win-A m2/m1
    // gearbox on the brick's integer_core_lmul. REUSES the tq2_0 ternary scaffold at C2
    // marginal cost, differing ONLY in the base-3 unpack.
    {
      bool hasTq10Core = false;
      loopBody.getBody().walk([&](weftrvv::GgmlBlockDotTQ10Q8KTernaryCoreOp) {
        hasTq10Core = true;
      });
      if (hasTq10Core)
        return emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10(
            rewriter, loc, scope, avlArg, sizeType, valueMap, loopBody);
    }
    // ---- Region walk (identify, no emit): the iq1_s grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ1SQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ1SQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq1_s grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the
    // loop induction variable (region arg 0) -- the anti-bypass tie (the emit
    // provably tracks the region content, not merely the loop op attrs). ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq1_s super-block scalar-accumulator grid body requires the "
                    "iq1_s ternary-grid integer-core brick + the single scalar "
                    "yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq1_s super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq1_s super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq1_s super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq1_s super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP
    // OP (the byte-exact schedule shape knobs); the per-region byte offsets + the
    // sub-block shape come off the iq1_s grid-core BRICK that owns them (I4 mirror).
    // The fp16 x.d @0 / fp32 y.d @0 are FIXED block_iq1_s / block_q8_K constants of
    // the emitter-inlined scalar fold (the grid core carries no d offset). The fixed
    // 2048-entry iq1s_grid TERNARY codebook is keyed off the brick op identity. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     //  50
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = 0;                                  //   0 (fp16 x.d)
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   2
    int64_t qhOffset = coreOp.getWeightQhByteOffset();          //  34
    int64_t activationDOffset = 0;                              //   0 (fp32 y.d)
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t bsumsOffset = coreOp.getActivationBsumsByteOffset();// 260
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t groupsPerSub = 4;  // 4 grid groups per sub-block (l=0..3)

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 2048-entry TERNARY grid codebook, keyed off the grid-core brick op
    // identity (the grid is NOT carried in the IR) from the canonical kIQ1SGrid --
    // byte-identical to the monolith's carried-attr decl.
    emitIQ1SCanonicalGridTableDecl(rewriter, loc);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region
    // arg `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args, so
    // the fold mutates it in place across iterations). NO 8-lane `sums` vector.
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // const uint64_t *grid = weft_iq1s_grid;  (the u64 grid base; the vluxei16
    // gather reads it as (const int64_t *) with u16 byte-offset indices idx*8.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type u64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u64PtrType, "weft_iq1s_grid");

    // The context the shared per-super-block grid body reads.
    IQ1SGridBodyContext cx{opName,        role,              sizeType,
                           weightPtrType, activationPtrType, weightDOffset,
                           qsOffset,      qhOffset,          activationDOffset,
                           q8Offset,      bsumsOffset,       subBlock,
                           numSubBlocks,  groupsPerSub,      gridArrayName};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core
      // brick's (base operand, block_index operand) through a per-iteration memo.
      // Correct wiring (weight base -> ABI weight, activation base -> ABI
      // activation, block_index == the IV) collapses to exactly TWO emitted bases
      // -- xb (super_block_base_x) and yb (super_block_base_y) -- byte-identical to
      // the monolith's blockBaseValue. A CHANGED brick base operand keys a
      // DIFFERENT memo entry and emits a DIFFERENT base (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact
      // order (weight x before activation y), from the grid-core brick that owns
      // both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq1_s per-super-block TERNARY-grid body (fp16*fp32 d fold scale,
      // qs/qh/q8/bsums bases, the two SCALAR states sumi + sumi1 via the vluxei16
      // grid gather, then the scalar delta fold into `sumf`). The SAME byte-exact
      // helper the (now-retired) monolith called.
      emitIQ1SSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq1_s super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);
    return mlir::success();
  }

// M-FLAT iq1_m super-block SCALAR-accumulator GRID emitter (the flip lowering, iq1_s
// SIBLING). The iq1_m branch of the fold_model "scalar_delta_grid" path (dispatched
// from emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an iq1_m
// grid-core brick). Same wrapper as the iq1_s emitter -- the `static const uint64_t
// weft_iq1m_grid[2048]` TERNARY grid decl (keyed off the iq1_m grid-core brick op
// identity from the canonical kIQ1MGrid), the `sumf` float SCALAR accumulator seeded
// ONCE outside the loop, nb = n / QK_K, the `weft_iq1m_grid` base literal, the outer
// emitc.for over nb, and (post-loop) the `*s` store -- delegating the in-loop
// per-super-block body to the SHARED emitIQ1MSuperBlockGridBody helper (the same one
// the retired monolith emitIQ1MQ8KBlockDot called), sourcing the per-super-block
// ADDRESSES from the grid-core brick's (base, block_index) OPERANDS via a per-body memo
// (anti-bypass W4), so the emitted C is byte-identical to the retired monolith by
// construction (same grid decl, same body helper, same facts, same order) modulo the
// source-op provenance token + the func name. iq1_m REUSES the whole iq1_s scaffold;
// the ONLY delta is the distinct brick + the iq1_m body anchor (the C2 marginal cost).
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBodyIq1M(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the iq1_m grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ1MQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ1MQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq1_m grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the
    // loop induction variable (region arg 0) -- the anti-bypass tie. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq1_m super-block scalar-accumulator grid body requires the "
                    "iq1_m ternary-grid integer-core brick + the single scalar "
                    "yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq1_m super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq1_m super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq1_m super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq1_m super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP
    // OP; the per-region byte offsets + the sub-block shape come off the iq1_m
    // grid-core BRICK that owns them (I4 mirror). iq1_m has NO fp16 weight d (the
    // scale is RECONSTRUCTED from scales[]) and reads NO bsums; the packed
    // iq1m_scale reconstruct + fp32 y.d @0 are FIXED constants of the emitter-inlined
    // scalar fold. The fixed 2048-entry iq1s_grid TERNARY codebook is keyed off the
    // brick op identity. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     //  56
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   0
    int64_t qhOffset = coreOp.getWeightQhByteOffset();          //  32
    int64_t scalesOffset = coreOp.getWeightScalesByteOffset();  //  48
    int64_t activationDOffset = 0;                              //   0 (fp32 y.d)
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t groupsPerSub = 4;  // 4 grid groups per sub-block (l=0..3)

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 2048-entry TERNARY grid codebook, keyed off the grid-core brick op
    // identity (the grid is NOT carried in the IR) from the canonical kIQ1MGrid --
    // byte-identical to the monolith's carried-attr decl (SAME literals as iq1_s,
    // distinct decl name `weft_iq1m_grid`).
    emitIQ1MCanonicalGridTableDecl(rewriter, loc);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator. NO 8-lane vector.
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // const uint64_t *weft_iq1m_grid;  (the u64 grid table name; cast to a
    // (const int64_t *) base for the vluxei16 indexed gather inside the loop.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_base"));
    mlir::Type u64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u64PtrType, "weft_iq1m_grid");

    // The context the shared per-super-block iq1_m grid body reads.
    IQ1MGridBodyContext cx{opName,        role,              sizeType,
                           weightPtrType, activationPtrType, qsOffset,
                           qhOffset,      scalesOffset,      activationDOffset,
                           q8Offset,      subBlock,          numSubBlocks,
                           groupsPerSub,  gridArrayName};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core
      // brick's (base operand, block_index operand) through a per-iteration memo
      // (anti-bypass -- a CHANGED brick base operand keys a DIFFERENT base).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // The two canonical bases in the monolith's byte-exact order (weight x before
      // activation y), from the grid-core brick that owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq1_m per-super-block TERNARY-grid body (the packed iq1m_scale fp16
      // reconstruct + fp32 d fold, the qs/qh/sc/q8 bases, the two SCALAR states
      // sumi1 + sumi2 via the per-half vluxei16 grid gather + the per-group Σq8 delta,
      // then the scalar delta fold into `sumf`). The SAME byte-exact helper the
      // (now-retired) monolith called.
      emitIQ1MSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq1_m super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);
    return mlir::success();
  }

// iq3_xxs super-block SCALAR-accumulator GRID-of-4 emitter (the flip lowering, iq1_s
// grid SIBLING). The iq3_xxs branch of the fold_model "scalar_delta_grid" path
// (dispatched from emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an
// iq3_xxs grid-core brick). It emits the wrapper -- the `weft_iq3xxs_grid`/`
// weft_iq3xxs_ksigns`/`weft_iq3xxs_kmask` decls, the `sumf` float SCALAR accumulator
// seeded once OUTSIDE the loop, nb = n / QK_K, the ONCE 8-lane kmask load + the (const
// int32_t *) grid32 view, the outer emitc.for over nb, the per-super-block base built
// from the brick's (base, block_index) via a shared memo (anti-bypass W4), and the
// trailing `*s = 0.25f*sumf` store -- delegating the in-loop per-super-block body to the
// SHARED emitIQ3XXSSuperBlockGridBody anchor (the same one the retired monolith called),
// so the emitted C is byte-identical to the retired monolith by construction (same
// decls, same body helper, same facts, same order) modulo the source-op provenance token
// + the func name.
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the iq3_xxs grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq3_xxs grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the loop
    // induction variable (region arg 0) -- the anti-bypass tie. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq3_xxs super-block scalar-accumulator grid body requires the "
                    "iq3_xxs GRID-of-4 integer-core brick + the single scalar "
                    "yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq3_xxs super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq3_xxs super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq3_xxs super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq3_xxs super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP OP
    // (the byte-exact schedule shape knobs); the per-region byte offsets + the
    // sub-block shape come off the iq3_xxs grid-core BRICK that owns them (I4 mirror).
    // The fp16 x.d @0 / fp32 y.d @0 are FIXED block_iq3_xxs / block_q8_K constants of
    // the emitter-inlined fold. The fixed 256-entry iq3xxs_grid GRID-of-4 codebook + the
    // 128-entry ksigns_iq2xs sign plane are keyed off the brick op identity. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     //  98
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = coreOp.getWeightDByteOffset();      //   0
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   2
    int64_t gasOffset = coreOp.getWeightGasByteOffset();        //  66
    int64_t activationDOffset = coreOp.getActivationDByteOffset();//  0
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t numGroups = 4;      // 4 sign groups per sub-block
    int64_t indicesPerSubBlock = 8; // 8 grid index bytes per sub-block (2 per group)
    int64_t groupLanes = 8;     // 8 grid lanes per sign group

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 256-entry GRID-of-4 codebook + the 128-entry ksigns sign plane, keyed off the
    // grid-core brick op identity (NOT carried in the IR) from the canonical
    // kIQ3XXSGrid / kIQ3XXSKsigns -- byte-identical to the monolith's carried-attr decls.
    emitIQ3XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ3XXSCanonicalKsignsTableDecl(rewriter, loc);
    // The kmask sign-bit selector {1<<j} is an inline const (trivial bit-position
    // vector), emitted as a decl + broadcast load ONCE above the super-block loop.
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t weft_iq3xxs_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region arg
    // `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args, so the
    // fold mutates it in place across iterations). NO 8-lane `sums` vector.
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // vuint8m1_t kmask = vle8(weft_iq3xxs_kmask, 8);  (ONCE) -- the FULL 8-bit selector
    // {1,2,4,8,16,32,64,128} the 8-lane group body masks the signs byte with.
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, "m1", "u8");
    mlir::Value kmask = emitOpaqueCallBuilt(
        rewriter, loc, u8CoreType, u8LoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value kmaskName = rewriter.create<emitc::LiteralOp>(
              loc, u8PtrType, "weft_iq3xxs_kmask");
          return {kmaskName, sizeLit(groupLanes)};
        },
        llvm::StringRef("kmask_table_load"));

    // const int32_t *grid32 = (const int32_t *)weft_iq3xxs_grid;  (signed-i32 view of
    // the uint32[256] grid for the vluxei16 indexed gather.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_i32_view"));
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type i32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int32_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u32PtrType, "weft_iq3xxs_grid");
    mlir::Value grid32 =
        rewriter.create<emitc::CastOp>(loc, i32PtrType, gridArrayName)
            .getResult();

    // The context the shared per-super-block grid-of-4 body reads.
    IQ3XXSGridBodyContext cx{opName,        role,              sizeType,
                             weightPtrType, activationPtrType, weightDOffset,
                             qsOffset,      gasOffset,         activationDOffset,
                             q8Offset,      subBlock,          numSubBlocks,
                             numGroups,     indicesPerSubBlock, grid32,
                             kmask};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core brick's
      // (base operand, block_index operand) through a per-iteration memo. Correct wiring
      // collapses to exactly TWO emitted bases -- xb (super_block_base_x) and yb
      // (super_block_base_y) -- byte-identical to the monolith's blockBaseValue. A
      // CHANGED brick base operand keys a DIFFERENT memo entry (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact order
      // (weight x before activation y), from the grid-core brick that owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq3_xxs per-super-block GRID-of-4 body (fp16*fp32 d fold scale, the
      // q3/gas/q8 bases, the aux32 4-bit-scale + 4-sign-group ksigns decode, the
      // vluxei16_v_i32m1 grid-of-4 gather + signed widening dot + bsum fold, then
      // `sumf += d*(float)bsum`). The SAME byte-exact helper the retired monolith called.
      emitIQ3XXSSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = 0.25f * sumf;  (the iq3_xxs trailing 1/4 factor -- a SEPARATE statement
    // OUTSIDE the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq3_xxs super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value oneQuarter =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.25f");
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, oneQuarter, sumfFinal)
            .getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), scaled);
    return mlir::success();
  }

// ---------------------------------------------------------------------------
// iq3_s super-block SCALAR-accumulator GRID-of-4 EXPLICIT-SIGNS emitter (the flip
// lowering, iq3_xxs grid SIBLING). The iq3_s branch of the fold_model
// "scalar_delta_grid" path (dispatched from emitTypedSuperBlockScalarDeltaGridLoopBody
// when the region carries an iq3_s grid-core brick). It emits the wrapper -- the
// `weft_iq3s_grid` GRID-of-4 decl + the inline `weft_iq3s_kmask[8]` decl (iq3_s has NO
// ksigns plane), the `sumf` float SCALAR accumulator seeded once OUTSIDE the loop, nb = n
// / QK_K, the ONCE 8-lane kmask load + the (const int32_t *) grid32 view, the outer
// emitc.for over nb, the per-super-block base built from the brick's (base, block_index)
// via a shared memo (anti-bypass W4), and the trailing `*s = sumf` store (NO trailing
// factor -- iq3_s applies none) -- delegating the in-loop per-super-block body to the
// SHARED emitIQ3SSuperBlockGridBody anchor (the same one the retired monolith called), so
// the emitted C is byte-identical to the retired monolith by construction (same decls,
// same body helper, same facts, same order) modulo the source-op provenance token + the
// func name.
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBodyIq3s(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the iq3_s grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ3SQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ3SQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq3_s grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the loop
    // induction variable (region arg 0) -- the anti-bypass tie. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq3_s super-block scalar-accumulator grid body requires the "
                    "iq3_s GRID-of-4 integer-core brick + the single scalar "
                    "yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq3_s super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq3_s super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq3_s super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq3_s super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP OP
    // (the byte-exact schedule shape knobs); the per-region byte offsets + the
    // sub-block shape come off the iq3_s grid-core BRICK that owns them (I4 mirror).
    // The fp16 x.d @0 / fp32 y.d @0 are FIXED block_iq3_s / block_q8_K constants of the
    // emitter-inlined fold. The fixed 512-entry iq3s_grid GRID-of-4 codebook is keyed off
    // the brick op identity (iq3_s has NO ksigns plane). ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     // 110
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = coreOp.getWeightDByteOffset();      //   0
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   2
    int64_t qhOffset = coreOp.getWeightQhByteOffset();          //  66
    int64_t signsOffset = coreOp.getWeightSignsByteOffset();    //  74
    int64_t scalesOffset = coreOp.getWeightScalesByteOffset();  // 106
    int64_t activationDOffset = coreOp.getActivationDByteOffset();//  0
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t numGroups = 4;      // 4 sign groups per sub-block
    int64_t indicesPerSubBlock = 8; // 8 grid index bytes per sub-block (2 per group)
    int64_t signsPerSubBlock = 4;   // 4 explicit sign bytes per sub-block
    int64_t groupLanes = 8;     // 8 grid lanes per sign group

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 512-entry GRID-of-4 codebook, keyed off the grid-core brick op identity (NOT
    // carried in the IR) from the canonical kIQ3SGrid -- byte-identical to the monolith's
    // carried-attr decl. iq3_s has NO ksigns plane (the signs are an explicit memory
    // region); the kmask sign-bit selector {1<<j} is an inline const.
    emitIQ3SCanonicalGridTableDecl(rewriter, loc);
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t weft_iq3s_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region arg
    // `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args, so the fold
    // mutates it in place across iterations). NO 8-lane `sums` vector.
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // vuint8m1_t kmask = vle8(weft_iq3s_kmask, 8);  (ONCE) -- the FULL 8-bit selector
    // {1,2,4,8,16,32,64,128} the 8-lane group body masks the explicit signs byte with.
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, "m1", "u8");
    mlir::Value kmask = emitOpaqueCallBuilt(
        rewriter, loc, u8CoreType, u8LoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value kmaskName = rewriter.create<emitc::LiteralOp>(
              loc, u8PtrType, "weft_iq3s_kmask");
          return {kmaskName, sizeLit(groupLanes)};
        },
        llvm::StringRef("kmask_table_load"));

    // const int32_t *grid32 = (const int32_t *)weft_iq3s_grid;  (signed-i32 view of the
    // uint32[512] grid for the vluxei16 indexed gather.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_i32_view"));
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type i32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int32_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u32PtrType, "weft_iq3s_grid");
    mlir::Value grid32 =
        rewriter.create<emitc::CastOp>(loc, i32PtrType, gridArrayName)
            .getResult();

    // The context the shared per-super-block grid-of-4 body reads.
    IQ3SGridBodyContext cx{opName,        role,              sizeType,
                           weightPtrType, activationPtrType, weightDOffset,
                           qsOffset,      qhOffset,          signsOffset,
                           scalesOffset,  activationDOffset, q8Offset,
                           subBlock,      numSubBlocks,      numGroups,
                           indicesPerSubBlock, signsPerSubBlock, grid32,
                           kmask};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core brick's
      // (base operand, block_index operand) through a per-iteration memo. Correct wiring
      // collapses to exactly TWO emitted bases -- xb (super_block_base_x) and yb
      // (super_block_base_y) -- byte-identical to the monolith's blockBaseValue. A CHANGED
      // brick base operand keys a DIFFERENT memo entry (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact order
      // (weight x before activation y), from the grid-core brick that owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq3_s per-super-block GRID-of-4 body (fp16*fp32 d fold scale, the
      // qs/qh/sgn/sc/q8 bases, the explicit two-nibble scale ls + qh 9th-bit inject +
      // explicit signs decode, the vluxei16_v_i32m1 grid-of-4 gather + signed widening
      // dot + bsum fold, then `sumf += d*(float)bsum`). The SAME byte-exact helper the
      // retired monolith called.
      emitIQ3SSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = sumf;  (iq3_s applies NO trailing factor -- a SEPARATE statement OUTSIDE the
    // accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq3_s super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);
    return mlir::success();
  }

// ---------------------------------------------------------------------------
// iq2_xxs super-block SCALAR-accumulator GRID-of-8 loop-body lowering (iq1_s grid
// SIBLING, SIGN-PLANE signs64 variant). The iq2_xxs branch of the fold_model
// "scalar_delta_grid" path (dispatched from emitTypedSuperBlockScalarDeltaGridLoopBody
// when the region carries an iq2_xxs grid-core brick). It emits the wrapper -- the
// `weft_iq2xxs_grid` GRID-of-8 decl + the DERIVED `weft_iq2xxs_signs64` signs64 sign
// plane decl, the `sumf` float SCALAR accumulator seeded once OUTSIDE the loop, nb = n /
// QK_K, the ONCE (const int64_t *) grid64 view + the (const int64_t *) signs64 view, the
// outer emitc.for over nb, the per-super-block base built from the brick's (base,
// block_index) via a shared memo (anti-bypass W4), and the trailing `*s = 0.125f*sumf`
// store -- delegating the in-loop per-super-block body to the SHARED
// emitIQ2XXSSuperBlockGridBody anchor (the same one the retired monolith called), so the
// emitted C is byte-identical to the retired monolith by construction (same decls, same
// body helper, same facts, same order) modulo the source-op provenance token + the func
// name. The Win-A gearbox is preserved: coreLmul reads the brick's integer_core_lmul
// (default m2), so the m2->m1 VLEN selection stamps the SAME shape onto the brick.
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the iq2_xxs grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq2_xxs grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the loop
    // induction variable (region arg 0) -- the anti-bypass tie. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq2_xxs super-block scalar-accumulator grid body requires the "
                    "iq2_xxs GRID-of-8 integer-core brick + the single scalar "
                    "yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq2_xxs super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq2_xxs super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq2_xxs super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq2_xxs super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP OP
    // (the byte-exact schedule shape knobs); the per-region byte offsets + the sub-block
    // shape come off the iq2_xxs grid-core BRICK that owns them (I4 mirror). The fp16 x.d
    // @0 / fp32 y.d @0 are FIXED block_iq2_xxs / block_q8_K constants of the
    // emitter-inlined fold. The fixed 256-entry iq2xxs_grid GRID-of-8 codebook + the
    // DERIVED keven_signs_q2xs signs64 sign plane are keyed off the brick op identity. The
    // Win-A coreLmul gearbox is read off the brick's integer_core_lmul (default m2). ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     //  66
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = coreOp.getWeightDByteOffset();      //   0
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   2
    int64_t activationDOffset = coreOp.getActivationDByteOffset();//  0
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t numGroups = 4;      // 4 grid/sign groups per sub-block
    llvm::StringRef coreLmul = coreOp.getIntegerCoreLmul().value_or("m2");

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 256-entry GRID-of-8 codebook + the DERIVED signs64 sign plane, keyed off the
    // grid-core brick op identity (NOT carried in the IR) from the canonical kIQ2XXSGrid /
    // kIQ2XXSKsigns -- byte-identical to the monolith's carried-attr / derived decls.
    emitIQ2XXSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XXSCanonicalSigns64TableDecl(rewriter, loc);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region arg
    // `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args, so the fold
    // mutates it in place across iterations). NO 8-lane `sums` vector.
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // const int64_t *grid64 = (const int64_t *)weft_iq2xxs_grid;  and
    // const int64_t *signs64 = (const int64_t *)weft_iq2xxs_signs64;  -- the i64 views
    // for the vluxei16 indexed gathers (ggml's grid64/signs64). Each grid u64 entry holds
    // 8 int8 grid bytes; each signs64 u64 entry holds the 8 +-1 sign bytes for that
    // selector.
    mlir::Type i64PtrViewType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type i8SignsPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_i64_view"));
    mlir::Value gridName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrViewType, "weft_iq2xxs_grid");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "signs_table_i64_view"));
    mlir::Value signsArrayName = rewriter.create<emitc::LiteralOp>(
        loc, i8SignsPtrType, "weft_iq2xxs_signs64");
    mlir::Value signs64 =
        rewriter.create<emitc::CastOp>(loc, i64PtrViewType, signsArrayName)
            .getResult();

    // The context the shared per-super-block grid-of-8 body reads.
    IQ2XXSGridBodyContext cx{opName,        role,              sizeType,
                             weightPtrType, activationPtrType, weightDOffset,
                             qsOffset,      activationDOffset, q8Offset,
                             subBlock,      numSubBlocks,      numGroups,
                             coreLmul,      gridName,          signs64};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core brick's
      // (base operand, block_index operand) through a per-iteration memo. Correct wiring
      // collapses to exactly TWO emitted bases -- xb (super_block_base_x) and yb
      // (super_block_base_y) -- byte-identical to the monolith's blockBaseValue. A CHANGED
      // brick base operand keys a DIFFERENT memo entry (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact order
      // (weight x before activation y), from the grid-core brick that owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq2_xxs per-super-block GRID-of-8 body (fp16*fp32 d fold scale, the
      // qs/q8 bases, the aux1 4-bit-scale + 4-sign-group decode, the TWO
      // vluxei16_v_i64<core> grid64/signs64 gathers + the vmul-onto-grid sign fold +
      // signed widening dot + bsum fold, then `sumf += d*(float)bsum`). The SAME
      // byte-exact helper the retired monolith called.
      emitIQ2XXSSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = 0.125f * sumf;  (the iq2_xxs trailing 1/8 factor -- a SEPARATE statement
    // OUTSIDE the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq2_xxs super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value oneEighth =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f");
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, oneEighth, sumfFinal)
            .getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), scaled);
    return mlir::success();
  }

// iq2_xs super-block SCALAR-accumulator per-half-scale GRID loop-body lowering (iq2_xxs
// grid SIBLING, SIGN-PLANE signs64 variant, PER-HALF explicit scale). The iq2_xs branch of
// the fold_model "scalar_delta_grid" path (dispatched from
// emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an iq2_xs grid-core
// brick). It emits the wrapper -- the `weft_iq2xs_grid[512]` decl + the DERIVED
// `weft_iq2xs_signs64` signs64 sign plane decl, the `sumf` float SCALAR accumulator seeded
// once OUTSIDE the loop, nb = n / QK_K, the ONCE (const int64_t *) grid64 view + the
// (const int64_t *) signs64 view, the outer emitc.for over nb, the per-super-block base
// built from the brick's (base, block_index) via a shared memo (anti-bypass W4), and the
// trailing `*s = 0.125f*sumf` store -- delegating the in-loop per-super-block body to the
// SHARED emitIQ2XSSuperBlockGridBody anchor, so the emitted C is byte-identical to the
// retired monolith by construction (same decls, same body helper, same facts, same order)
// modulo the source-op provenance token + the func name. UNLIKE iq2_xxs there is NO gearbox
// (fixed 16-lane per-half shape).
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xs(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the iq2_xs grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ2XSQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ2XSQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq2_xs grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the loop
    // induction variable (region arg 0) -- the anti-bypass tie. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq2_xs super-block scalar-accumulator grid body requires the "
                    "iq2_xs per-half-scale GRID integer-core brick + the single "
                    "scalar yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq2_xs super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq2_xs super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq2_xs super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq2_xs super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP OP
    // (the byte-exact schedule shape knobs); the per-region byte offsets + the sub-block
    // shape come off the iq2_xs grid-core BRICK that owns them (I4 mirror). The fp16 x.d
    // @0 / fp32 y.d @0 are FIXED block_iq2_xs / block_q8_K constants of the
    // emitter-inlined fold. The fixed 512-entry iq2xs_grid codebook + the DERIVED
    // keven_signs_q2xs signs64 sign plane are keyed off the brick op identity. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     //  74
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = coreOp.getWeightDByteOffset();      //   0
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   2
    int64_t scalesOffset = coreOp.getWeightScalesByteOffset();  //  66
    int64_t activationDOffset = coreOp.getActivationDByteOffset();//  0
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t numGroupsPerHalf = 2;      // 2 grid/sign groups per 16-lane half

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 512-entry GRID codebook + the DERIVED signs64 sign plane, keyed off the grid-core
    // brick op identity (NOT carried in the IR) from the canonical kIQ2XSGrid / kIQ2XSKsigns
    // -- byte-identical to the monolith's carried-attr / derived decls.
    emitIQ2XSCanonicalGridTableDecl(rewriter, loc);
    emitIQ2XSCanonicalSigns64TableDecl(rewriter, loc);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region arg
    // `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args, so the fold
    // mutates it in place across iterations).
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // const int64_t *grid64 = (const int64_t *)weft_iq2xs_grid;  and
    // const int64_t *signs64 = (const int64_t *)weft_iq2xs_signs64;  -- the i64 views for
    // the vluxei16 indexed gathers. weft_iq2xs_grid is already int64_t[512] (the literal IS
    // an int64_t*); weft_iq2xs_signs64 is int8_t[1024] re-cast to const int64_t*.
    mlir::Type i64PtrViewType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type i8SignsPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_i64_view"));
    mlir::Value gridName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrViewType, "weft_iq2xs_grid");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "signs_table_i64_view"));
    mlir::Value signsArrayName = rewriter.create<emitc::LiteralOp>(
        loc, i8SignsPtrType, "weft_iq2xs_signs64");
    mlir::Value signs64 =
        rewriter.create<emitc::CastOp>(loc, i64PtrViewType, signsArrayName)
            .getResult();

    // The context the shared per-super-block per-half-scale grid body reads.
    IQ2XSGridBodyContext cx{opName,        role,              sizeType,
                            weightPtrType, activationPtrType, weightDOffset,
                            qsOffset,      scalesOffset,      activationDOffset,
                            q8Offset,      subBlock,          numSubBlocks,
                            numGroupsPerHalf, gridName,       signs64};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core brick's
      // (base operand, block_index operand) through a per-iteration memo. Correct wiring
      // collapses to exactly TWO emitted bases -- xb (super_block_base_x) and yb
      // (super_block_base_y) -- byte-identical to the monolith's blockBaseValue. A CHANGED
      // brick base operand keys a DIFFERENT memo entry (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact order
      // (weight x before activation y), from the grid-core brick that owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq2_xs per-super-block per-half-scale GRID body (fp16*fp32 d fold scale,
      // the qs/sc/q8 bases, the per-sub-block explicit 4-bit scale ls1/ls2, the two-half
      // 16-lane decode: the TWO vluxei16_v_i64m1 grid64/signs64 gathers + the vmul-onto-grid
      // sign fold + signed widening dot + bsum fold, then `sumf += d*(float)bsum`). The SAME
      // byte-exact helper the retired monolith called.
      emitIQ2XSSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = 0.125f * sumf;  (the iq2_xs trailing 1/8 factor -- a SEPARATE statement OUTSIDE
    // the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq2_xs super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value oneEighth =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f");
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, oneEighth, sumfFinal)
            .getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), scaled);
    return mlir::success();
  }

// iq2_s super-block SCALAR-accumulator per-half-scale GRID loop-body lowering (iq2_xs grid
// SIBLING, SIGN-PLANE explicit-signs variant, PER-HALF explicit scale). The iq2_s branch of
// the fold_model "scalar_delta_grid" path (dispatched from
// emitTypedSuperBlockScalarDeltaGridLoopBody when the region carries an iq2_s grid-core
// brick). It emits the wrapper -- the `weft_iq2s_grid[1024]` decl + the UNIVERSAL
// `weft_iq2s_signs256` sign plane decl, the `sumf` float SCALAR accumulator seeded once
// OUTSIDE the loop, nb = n / QK_K, the ONCE (const int64_t *) grid64 view + the
// (const int64_t *) signs256 view, the outer emitc.for over nb, the per-super-block base
// built from the brick's (base, block_index) via a shared memo (anti-bypass W4), and the
// trailing `*s = 0.125f*sumf` store -- delegating the in-loop per-super-block body to the
// SHARED emitIQ2SSuperBlockGridBody anchor, so the emitted C is byte-identical to the
// retired monolith by construction (same decls, same body helper, same facts, same order)
// modulo the source-op provenance token + the func name. Like iq2_xs there is NO gearbox
// (fixed 16-lane per-half shape).
mlir::LogicalResult
VariantToEmitCFunc::emitTypedSuperBlockScalarDeltaGridLoopBodyIq2s(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    weftrvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) const {
    (void)scope;
    // ---- Region walk (identify, no emit): the iq2_s grid-core brick + yield. ----
    weftrvv::GgmlBlockDotIQ2SQ8KGridCoreOp coreOp;
    weftrvv::TypedSuperBlockBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotIQ2SQ8KGridCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedSuperBlockBlockDotLoopYieldOp>(
                       bodyOp))
        yieldOp = o;
    });

    // ---- Region-driven gate (fail-closed, I7): the byte-exact SCALAR-accumulator
    // GRID body requires the iq2_s grid-core brick + the SINGLE scalar yield, the
    // (index, sumf scalar) entry-arg pair, and the brick's block_index tied to the loop
    // induction variable (region arg 0) -- the anti-bypass tie. ----
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "iq2_s super-block scalar-accumulator grid body requires the "
                    "iq2_s per-half-scale GRID integer-core brick + the single "
                    "scalar yield");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "iq2_s super-block scalar-accumulator grid body region must "
                    "carry exactly the (super_block_index, sumf) pair");
    mlir::Value sbIndex = coreBlock.getArgument(0);
    mlir::Value sumfArg = coreBlock.getArgument(1);
    if (yieldOp.getSumsNext() != sumfArg || yieldOp.getSumfNext())
      return rewriter.notifyMatchFailure(
          yieldOp, "iq2_s super-block scalar yield must carry the loop-carried "
                   "sumf scalar ALONE (no second operand under the scalar fold)");
    if (coreOp.getBlockIndex() != sbIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the iq2_s super-block grid-core brick's block_index must be "
                    "the loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    // ---- ABI operands (the same three the monolith reads). ----
    mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq2_s super-block scalar-accumulator grid ABI operand unmapped");

    llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // ---- The block-format structural facts. The strides + qk come off the LOOP OP
    // (the byte-exact schedule shape knobs); the per-region byte offsets + the sub-block
    // shape come off the iq2_s grid-core BRICK that owns them (I4 mirror). The fp16 x.d
    // @0 / fp32 y.d @0 are FIXED block_iq2_s / block_q8_K constants of the
    // emitter-inlined fold. The fixed 1024-entry iq2s_grid codebook + the UNIVERSAL
    // signs256 sign plane are keyed off the brick op identity. ----
    int64_t qk = loopBody.getQk();                              // 256
    int64_t subBlock = coreOp.getSubBlock();                    //  32
    int64_t weightStride = loopBody.getWeightBlockStride();     //  82
    int64_t activationStride = loopBody.getActivationBlockStride(); // 292
    int64_t weightDOffset = coreOp.getWeightDByteOffset();      //   0
    int64_t qsOffset = coreOp.getWeightQsByteOffset();          //   2
    int64_t signsOffset = coreOp.getWeightSignsByteOffset();    //  34
    int64_t qhOffset = coreOp.getWeightQhByteOffset();          //  66
    int64_t scalesOffset = coreOp.getWeightScalesByteOffset();  //  74
    int64_t activationDOffset = coreOp.getActivationDByteOffset();//  0
    int64_t q8Offset = coreOp.getActivationQuantByteOffset();   //   4
    int64_t numSubBlocks = qk / subBlock;                       //   8
    int64_t groupsPerSub = 4;          // 4 grid groups per sub-block (l=0..3)
    int64_t numGroupsPerHalf = 2;      // 2 grid/sign groups per 16-lane half

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 1024-entry GRID codebook + the UNIVERSAL signs256 sign plane, keyed off the
    // grid-core brick op identity (NOT carried in the IR) from the canonical kIQ2SGrid /
    // the universal 8-bit-to-per-lane expansion -- byte-identical to the monolith's
    // carried-attr / inline decls.
    emitIQ2SCanonicalGridTableDecl(rewriter, loc);
    emitIQ2SCanonicalSigns256TableDecl(rewriter, loc);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator (the typed region arg
    // `sumf` lowers to this emitc.variable lvalue; emitc.for has no iter_args, so the fold
    // mutates it in place across iterations).
    (void)sumfArg;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // const int64_t *grid64 = (const int64_t *)weft_iq2s_grid;  and
    // const int64_t *signs256 = (const int64_t *)weft_iq2s_signs256;  -- the i64 views for
    // the vluxei16 indexed gathers. weft_iq2s_grid is already int64_t[1024] (the literal IS
    // an int64_t*); weft_iq2s_signs256 is int8_t[2048] re-cast to const int64_t*.
    mlir::Type i64PtrViewType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Type i8SignsPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_i64_view"));
    mlir::Value gridName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrViewType, "weft_iq2s_grid");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "signs_table_i64_view"));
    mlir::Value signsArrayName = rewriter.create<emitc::LiteralOp>(
        loc, i8SignsPtrType, "weft_iq2s_signs256");
    mlir::Value signs256 =
        rewriter.create<emitc::CastOp>(loc, i64PtrViewType, signsArrayName)
            .getResult();

    // The context the shared per-super-block per-half-scale grid body reads.
    IQ2SGridBodyContext cx{opName,        role,              sizeType,
                           weightPtrType, activationPtrType, weightDOffset,
                           qsOffset,      signsOffset,       qhOffset,
                           scalesOffset,  activationDOffset, q8Offset,
                           subBlock,      numSubBlocks,      groupsPerSub,
                           numGroupsPerHalf, gridName,       signs256};

    // ---- The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1). ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // W-E: per-super-block base = base + ib*stride, built from the grid-core brick's
      // (base operand, block_index operand) through a per-iteration memo. Correct wiring
      // collapses to exactly TWO emitted bases -- xb (super_block_base_x) and yb
      // (super_block_base_y) -- byte-identical to the monolith's blockBaseValue. A CHANGED
      // brick base operand keys a DIFFERENT memo entry (anti-bypass).
      llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
          blockBaseMemo;
      auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                              int64_t stride, const char *step) -> mlir::Value {
        std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
        auto it = blockBaseMemo.find(key);
        if (it != blockBaseMemo.end())
          return it->second;
        mlir::Value emittedBase = valueMap.lookup(bufferSSA);
        rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };

      // Establish the two canonical bases up-front in the monolith's byte-exact order
      // (weight x before activation y), from the grid-core brick that owns both bases.
      mlir::Value xb =
          blockBaseFor(coreOp.getWeightBase(), coreOp.getBlockIndex(),
                       weightStride, "super_block_base_x");
      mlir::Value yb =
          blockBaseFor(coreOp.getActivationBase(), coreOp.getBlockIndex(),
                       activationStride, "super_block_base_y");

      // The shared iq2_s per-super-block per-half-scale GRID body (fp16*fp32 d fold scale,
      // the qs/sgn/qh/sc/q8 bases, the per-sub-block explicit 4-bit scale ls1/ls2, the qh
      // plane, the two-half 16-lane decode: the TWO vluxei16_v_i64m1 grid64/signs256 gathers
      // + the vmul-onto-grid sign fold + signed widening dot + bsum fold, then
      // `sumf += d*(float)bsum`). The SAME byte-exact helper the retired monolith called.
      emitIQ2SSuperBlockGridBody(
          rewriter, loc, cx, xb, yb,
          llvm::cast<mlir::TypedValue<emitc::LValueType>>(sumfVar.getResult()));
    }

    // *s = 0.125f * sumf;  (the iq2_s trailing 1/8 factor -- a SEPARATE statement OUTSIDE
    // the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(
          loopBody,
          "iq2_s super-block scalar-accumulator grid output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value oneEighth =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f");
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, oneEighth, sumfFinal)
            .getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), scaled);
    return mlir::success();
  }

// ---------------------------------------------------------------------------
// q2_K super-block SCALAR integer core (byte-exact anchor). Extracted from the
// retired monolith emitQ2_KQ8_KBlockDot so the front-door-constructed typed
// super-block SCALAR-accumulator loop lowers byte-identically. Emits ONE
// super-block's 2-bit weight unpack into aux8[256] + the nested 16-sub-block
// PLAIN-nibble scaled scalar i32 dot, returning the two per-super-block SCALAR
// integer-state lvalues (isum, summs). xb/yb are the per-super-block bases; aux8
// is function-scoped scratch declared by the caller.
std::pair<mlir::TypedValue<emitc::LValueType>,
          mlir::TypedValue<emitc::LValueType>>
VariantToEmitCFunc::emitQ2_KSuperBlockIntegerCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q2_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type i32Type = cx.i32Type;
    mlir::Type i32ImmType = cx.i32ImmType;
    mlir::Type u8m2Type = cx.u8m2Type;
    mlir::Type i8m2Type = cx.i8m2Type;
    mlir::Type i8m1Type = cx.i8m1Type;
    mlir::Type i16m2Type = cx.i16m2Type;
    mlir::Type i32m1Type = cx.i32m1Type;
    mlir::Type i8ElemType = cx.i8ElemType;
    mlir::Type i8PtrType = cx.i8PtrType;
    mlir::Type u8PtrType = cx.u8PtrType;
    mlir::Type constU8Type = cx.constU8Type;
    mlir::Type constI16Type = cx.constI16Type;
    mlir::Type constI16PtrType = cx.constI16PtrType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    int64_t qk = cx.qk;
    int64_t subBlock = cx.subBlock;
    int64_t numSubBlocks = cx.numSubBlocks;
    int64_t scalesOffset = cx.scalesOffset;
    int64_t qsOffset = cx.qsOffset;
    int64_t q8Offset = cx.q8Offset;
    int64_t bsumsOffset = cx.bsumsOffset;

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

    return {llvm::cast<mlir::TypedValue<emitc::LValueType>>(isumVar.getResult()),
            llvm::cast<mlir::TypedValue<emitc::LValueType>>(
                summsVar.getResult())};
  }

// The q2_K per-super-block SCALAR fp32 fold `sumf += dall*isum - dmin*summs`
// (byte-exact anchor). dy loaded once, dall = fp16(xb+80)*dy, dmin =
// fp16(xb+82)*dy, then the two products + sub + add as ONE emitc.expression.
void VariantToEmitCFunc::emitQ2_KScalarFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const Q2_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::LValueType> sumfVar,
    mlir::TypedValue<emitc::LValueType> isumVar,
    mlir::TypedValue<emitc::LValueType> summsVar) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type i32Type = cx.i32Type;
    mlir::Type floatType = cx.floatType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    llvm::StringRef fp16ReadCallee = cx.fp16ReadCallee;
    int64_t weightDOffset = cx.weightDOffset;
    int64_t weightDminOffset = cx.weightDminOffset;
    int64_t activationDOffset = cx.activationDOffset;

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

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
          rewriter.create<emitc::CastOp>(loc, floatType, isumFinal).getResult();
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
          rewriter.create<emitc::SubOp>(loc, floatType, posProduct, minProduct);
      // sumf + (dall*isum - dmin*summs)  -- the `+=`.
      mlir::Value sumfNext =
          rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, delta);
      rewriter.create<emitc::YieldOp>(loc, sumfNext);
    }
    rewriter.create<emitc::VerbatimOp>(loc,
                                       assignComment("sumf", opName, role));
    rewriter.create<emitc::AssignOp>(loc, sumfVar, foldExpr.getResult());
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
