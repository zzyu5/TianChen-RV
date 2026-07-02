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

// VariantToEmitCFunc codebook + FP4 emit methods: iq4_nl / iq4_xs (int8
// codebook) and mxfp4 / nvfp4 (FP4 micro-exponent). Split out of RVVToEmitC.cpp
// as a pure code move; the emitted C is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitIQ4NLQ8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: iq4_nl codebook_gather_nibble / half-block / SumiTimesScales
    // (fp16 weight scale) instance of the descriptor-driven emitFlatBlockDot --
    // the 2nd primitive class (Fork B codebook extension). Resolve the ABI
    // operands + provenance, enforce the codebook I7 fail-closed anchor guard,
    // derive the block-format descriptor (codebook + gather + fold from
    // `kind`/attrs) + the scheduled BlockDotFacts, emit the shared body.
    tcrvrvv::GgmlBlockDotIQ4NLQ80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ4NLQ80Op>(op))
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

    // I7 FAIL-CLOSED: an UN-scheduled (attr-less, no integer_core_lmul) codebook op
    // must NOT lower. The codebook gather indexes a broadcast 16-entry table; the
    // emitter's default m1 anchor has gather VLMAX < 16 below VLEN=128, so a high
    // nibble index would silently read 0. The materialize-schedule pass at a sub-128
    // tier leaves the op attr-less (no legal anchor exists -- the codebook class is
    // Zvl128b-gated) AND stamps NO minimum_vlen, so the VERIFIER cannot tell it from
    // the legal pre-schedule input. The lowering boundary CAN: every VLEN>=128 path
    // stamps a legal m1/mf2 first (V implies VLEN >= 128), so an op that reaches the
    // emitter WITHOUT an anchor can only be the unsafe sub-128 leftover. Refuse it.
    if (!blockDot.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          blockDot,
          "refusing to lower an UN-scheduled codebook op (no integer_core_lmul): "
          "the codebook class is Zvl128b-gated and the emitter's default m1 anchor "
          "cannot host the 16-entry gather below VLEN=128 (a nibble index >= VLMAX "
          "silently reads 0); the gearbox must stamp a legal anchor first "
          "(materialize-schedule on a VLEN>=128 target) -- fail-closed (I7)");

    std::optional<FlatBlockDotDescriptor> descriptor =
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-codebook");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitIQ4XSQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ4XSQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ4XSQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq4_xs block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq4_xs block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The codebook gather REQUIRES the m1 anchor (VLMAX >= 16 to index all 16
    // table entries); the verifier pins it. i8 source LMUL m1 -> i16 product m2.
    llvm::StringRef coreLmul = "m1";
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef wideLmul = wideningChain.l16;
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t subBlock = blockDot.getSubBlock();                      //  32
    int64_t weightStride = blockDot.getWeightBlockStride();         // 136
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t scalesHOffset = blockDot.getWeightScalesHByteOffset();  //   2
    int64_t scalesLOffset = blockDot.getWeightScalesLByteOffset();  //   4
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   8
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t halfBlock = subBlock / 2; // 16 nibble bytes / q8 half lanes per sub-block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto intLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, intType, std::to_string(v));
    };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constU16Type = emitc::OpaqueType::get(ctx, "const uint16_t");
    mlir::Type u16PtrType = emitc::PointerType::get(constU16Type);
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 16-entry non-linear codebook is a STRUCTURAL fact off the typed attr
    // (I4 mirror), the SAME kvalues_iq4nl[16] table iq4_nl uses. Emit it as a
    // `static const int8_t[16]` decl ONCE, then broadcast-load via vle8 ONCE above
    // the super-block loop (the decl renders the verified attr entries; the table
    // register is reused by every gather).
    llvm::ArrayRef<int8_t> codebook = blockDot.getCodebook();
    {
      std::string decl = "static const int8_t tcrv_iq4_xs_kvalues[16] = {";
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // float sumf = 0.0f;  (function-scoped accumulator across the super-block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // vint8m1_t values = __riscv_vle8_v_i8m1(tcrv_iq4_xs_kvalues, 16);  (the
    // codebook table broadcast into a vreg ONCE; reused by every gather).
    std::string tableLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    mlir::Value values = emitOpaqueCallBuilt(
        rewriter, loc, i8CoreType, tableLoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value tableName = rewriter.create<emitc::LiteralOp>(
              loc, i8PtrType, "tcrv_iq4_xs_kvalues");
          return {tableName, sizeLit(codebook.size())};
        },
        llvm::StringRef("codebook_table_load"));

    // Per-super-block base address arithmetic: xb = vx + ibl*136; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The codebook decode + asymmetric product + reduce for ONE sub-block (the
    // 16-nibble-byte half-block), seeded 0, returning the scalar sumi. This is
    // iq4_nl's elided core copied verbatim: load the 16 packed nibble bytes at
    // qs + j*16, split into the two UNSIGNED index lanes (vand 0x0F / vsrl 0x04),
    // GATHER each through the broadcast codebook table (vrgather_vv_i8m1), feed the
    // SAME emitOffsetBinaryProductFromDecodedValue chain (vwmul low <-> q8[0..15],
    // vwmacc + high <-> q8[16..31]) -> vwredsum -> scalar. iq4_nl's sumi already
    // equals _generic's (sumi1 + sumi2).
    auto emitSubBlockSumi =
        [&](mlir::Value qsBase, mlir::Value q8Base) -> mlir::FailureOr<mlir::Value> {
      // The codebook anchor is m1: one vsetvl_e8m1(16) covers the half-block.
      std::string innerSetvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
      mlir::Value vl = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(halfBlock)};
          });

      // Load the 16 packed nibble bytes (UNSIGNED) + the two q8 halves (SIGNED).
      std::string wLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      mlir::Value w = emitOpaqueCall(rewriter, loc, u8CoreType, wLoadCallee,
                                     mlir::ValueRange{qsBase, vl}, opName, role);
      std::string yLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, yLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      mlir::Value y0 = loadY(q8Base);
      mlir::Value q8HighPtr =
          rewriter.create<emitc::AddOp>(loc, i8PtrType, q8Base, sizeLit(16));
      mlir::Value y1 = loadY(q8HighPtr);

      // The codebook nibble decode: split into the two UNSIGNED index lanes, then
      // gather each through the broadcast table -> signed-i8 weight lanes v0/v1.
      auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                         llvm::StringRef amount) -> mlir::Value {
        std::string callee = ("__riscv_" + mnemonic + "_u8" + coreLmul).str();
        return emitOpaqueCallBuilt(
            rewriter, loc, u8CoreType, callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value amt = rewriter.create<emitc::LiteralOp>(loc, intType,
                                                                  amount.str());
              return {src, amt, vl};
            });
      };
      mlir::Value idxLow = u8ImmOp("vand_vx", w, "0x0F");
      mlir::Value idxHigh = u8ImmOp("vsrl_vx", w, "0x04");
      std::string gatherCallee = ("__riscv_vrgather_vv_i8" + coreLmul).str();
      auto gather = [&](mlir::Value idx) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, gatherCallee,
                              mlir::ValueRange{values, idx, vl}, opName, role);
      };
      mlir::Value v0 = gather(idxLow);
      mlir::Value v1 = gather(idxHigh);

      // The SAME asymmetric signed widening product iq4_nl uses (vwmul low <->
      // q8[0..15], vwmacc + high <-> q8[16..31]) -> i16 product.
      mlir::FailureOr<mlir::Value> product =
          emitOffsetBinaryProductFromDecodedValue(rewriter, loc, v0, v1, y0, y1,
                                                  vl, i16WideType, 16, wideLmul,
                                                  "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-sub-block scalar: seed lane0 = 0, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, seedCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zeroSeed =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zeroSeed, sizeLit(1)};
          });
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red = emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                                       mlir::ValueRange{*product, seed, vl},
                                       opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
    };

    // The outer super-block loop: for (size_t ibl = 0; ibl < nb; ibl += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ibl = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ibl, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ibl, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // d4d8 = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);
      // (the fp16 weight super-block scale times the fp32 q8_K activation scale,
      // computed ONCE per super-block, mirroring _generic's d4d8). The two scale
      // reads are separate ops; d4d8 is one mul.
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      mlir::Value dx = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                      mlir::ValueRange{dxAddr}, opName, role,
                                      llvm::StringRef("fcvt.s.h"));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_activation_d"));
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
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
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d4d8"));
      mlir::Value d4d8 =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // uint16_t scales_h = *(const uint16_t *)(xb + 2);  (loaded ONCE; the 8
      // per-sub-block high scale-bit pairs live here).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scales_h_load"));
      mlir::Value shAddr = xb;
      if (scalesHOffset != 0)
        shAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(scalesHOffset));
      mlir::Value shPtr =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, shAddr).getResult();
      mlir::Value shIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value shElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(shPtr),
                  shIndex0)
              .getResult();
      mlir::Value scalesH16 =
          rewriter.create<emitc::LoadOp>(loc, constU16Type, shElem).getResult();
      // int scales_h = (int)scales_h;  (zero-extend the uint16 so the shifts/ands
      // run in the int domain, exactly as _generic's `uint16_t h` does -- the high
      // bits beyond bit 15 are never touched: the largest shift is 2*7=14).
      mlir::Value scalesH =
          rewriter.create<emitc::CastOp>(loc, intType, scalesH16).getResult();

      // const uint8_t *scales_l = xb + 4;  (the 4 low scale-nibble bytes).
      mlir::Value slAddr = xb;
      if (scalesLOffset != 0)
        slAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(scalesLOffset));
      mlir::Value scalesLPtr =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, slAddr).getResult();

      // const uint8_t *qs = xb + 8;  const int8_t *q8 = yb + 4;  (the sub-block
      // bases; advanced by j*16 (qs) / j*32 (q8) per sub-block).
      mlir::Value qsBase0 = xb;
      if (qsOffset != 0)
        qsBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qsOffset));
      mlir::Value qsBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();

      // The FLAT per-sub-block loop (j = 0..7), fully unrolled so each sub-block's
      // SIGNED scale extraction + codebook dot + float fold is emitted in STRICT
      // ascending order (fp non-associativity byte-exact).
      auto bAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseAndOp>(loc, intType, a, b)
            .getResult();
      };
      auto bOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseOrOp>(loc, intType, a, b)
            .getResult();
      };
      auto bShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseRightShiftOp>(loc, intType, a, b)
            .getResult();
      };
      auto bShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, intType, a, b)
            .getResult();
      };
      for (int64_t j = 0; j < numSubBlocks; ++j) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_signed_scale"));
        // uint8_t sl = scales_l[j/2];  (the low scale-nibble byte for this pair).
        mlir::Value slIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(j / 2));
        mlir::Value slElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(scalesLPtr),
                    slIdx)
                .getResult();
        mlir::Value slU8 =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, slElem).getResult();
        mlir::Value sl =
            rewriter.create<emitc::CastOp>(loc, intType, slU8).getResult();
        // low  = (sl >> (4*(j%2))) & 0xf
        mlir::Value low = bAnd(bShr(sl, intLit(4 * (j % 2))), intLit(0xf));
        // high = ((scales_h >> (2*j)) & 0x3) << 4
        mlir::Value high =
            bShl(bAnd(bShr(scalesH, intLit(2 * j)), intLit(0x3)), intLit(4));
        // int ls = low | high;  (the unsigned 6-bit scale, [0,63]).
        mlir::Value ls = bOr(low, high);
        // float d1 = d4d8 * (float)(ls - 32);  -- a SEPARATE op (mirrors
        // _generic's named d1 = d4d8*(ls-32); the (ls-32) round and the d4d8
        // multiply stay separate from the sumi multiply so NO FMA fuses them).
        mlir::Value lsBias =
            rewriter.create<emitc::SubOp>(loc, intType, ls, intLit(32))
                .getResult();
        mlir::Value lsFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, lsBias).getResult();
        mlir::Value d1 =
            rewriter.create<emitc::MulOp>(loc, floatType, d4d8, lsFloat)
                .getResult();

        // The per-sub-block codebook integer dot (iq4_nl core, seed 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_codebook_dot"));
        mlir::Value qsPtr =
            (j == 0) ? qsBase
                     : rewriter
                           .create<emitc::AddOp>(loc, u8PtrType, qsBase,
                                                 sizeLit(j * halfBlock))
                           .getResult();
        mlir::Value q8Ptr =
            (j == 0) ? q8Base
                     : rewriter
                           .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                                 sizeLit(j * subBlock))
                           .getResult();
        mlir::FailureOr<mlir::Value> sumi = emitSubBlockSumi(qsPtr, q8Ptr);
        if (mlir::failed(sumi))
          return mlir::failure();

        // sumf = sumf + d1 * (float)sumi;  -- ONE emitc.expression so it renders
        // as ggml's single C statement and the compiler fuses the SAME FMA under
        // -ffp-contract=on/default. The emitc.load temps stay OUTSIDE the
        // expression. Invoked in STRICT ascending (super-block, sub-block) order.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "fp32_accumulate"));
        mlir::Value sumfCur =
            rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
        auto accumExpr = rewriter.create<emitc::ExpressionOp>(
            loc, floatType, /*do_not_inline=*/false);
        {
          mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
          mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
          rewriter.setInsertionPointToStart(exprBlock);
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, *sumi).getResult();
          mlir::Value blockTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, d1, sumiFloat)
                  .getResult();
          mlir::Value sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm)
                  .getResult();
          rewriter.create<emitc::YieldOp>(loc, sumfNext);
        }
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumf", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq4_xs block-dot output not a pointer");
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

// The structured E8M0 -> fp32 HALF weight scale (FlatWeightScaleSource::E8M0,
// the mxfp4 FP4-class scale source): GGML_E8M0_TO_FP32_HALF(e) = 2^(e-128),
// reconstructed by ggml's EXACT bit construction (no scalbnf/ldexpf). Read
// e = *(const uint8_t *)(xb), build uint32_t bits = (e < 2) ? (0x00200000u <<
// (e & 0x1F)) : ((e - 1) << 23), reinterpret as float via *(const float *)&bits.
// All structured emitc nodes; the emitted C is byte-identical to the former
// inline mxfp4 emitter. Called by emitFlatBlockDot's emitBlockCore when the
// descriptor's weightScaleSource is E8M0.
mlir::Value VariantToEmitCFunc::emitE8M0HalfScale(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, mlir::Value xb,
    llvm::StringRef opName, llvm::StringRef role) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    // uint8_t e = *(const uint8_t *)(xb);  -- the E8M0 exponent byte at +0.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "e8m0_exponent_load"));
    mlir::Value xbU8 =
        rewriter.create<emitc::CastOp>(loc, u8PtrType, xb).getResult();
    mlir::Value eIdx0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value eElem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc, mlir::cast<mlir::TypedValue<emitc::PointerType>>(xbU8),
                eIdx0)
            .getResult();
    mlir::Value e =
        rewriter.create<emitc::LoadOp>(loc, constU8Type, eElem).getResult();
    // uint32_t e32 = (uint32_t) e;  (the shifts run in the uint32_t domain).
    mlir::Value e32 =
        rewriter.create<emitc::CastOp>(loc, u32Type, e).getResult();
    // bits = (e < 2) ? (0x00200000u << (e & 0x1F)) : ((e - 1) << 23);
    // ggml's ggml_e8m0_to_fp32_half (ggml-impl.h) is `(e<2) ? (0x00200000<<e) :
    // ((e-1)<<23)`. Each branch op is a SEPARATE structured emitc statement here,
    // and both branch values are computed BEFORE the ternary select (the mlir-cpp
    // emitter renders emitc.conditional as a statement-level ternary, not an
    // inline short-circuit). So the denormal shift `0x00200000u << e` is evaluated
    // even for the NORMAL e>=2 case; with e in [2,255] a raw `<< e` is C UB
    // (shift count >= 32). MASK the denormal shift count with `& 0x1F` so it is
    // ALWAYS well-defined (< 32). This is byte-IDENTICAL to ggml: the masked value
    // is consumed ONLY when e < 2, where `e & 0x1F == e` exactly (e is 0 or 1), so
    // the denormal scales 2^-128 / 2^-127 are bit-for-bit preserved; for e >= 2 the
    // (now well-defined) denormal value is discarded by the select.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "e8m0_to_fp32_half_bits"));
    mlir::Value two = rewriter.create<emitc::LiteralOp>(loc, u32Type, "2");
    mlir::Value isDenorm =
        rewriter
            .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::lt, e32,
                                  two)
            .getResult();
    // denormal branch: 0x00200000u << (e & 0x1F)  (e in {0,1} -> 2^-128/2^-127;
    // the mask keeps the shift count well-defined for the discarded e>=2 path).
    mlir::Value denormBase =
        rewriter.create<emitc::LiteralOp>(loc, u32Type, "0x00200000u");
    mlir::Value shiftMask =
        rewriter.create<emitc::LiteralOp>(loc, u32Type, "0x1F");
    mlir::Value denormShift =
        rewriter.create<emitc::BitwiseAndOp>(loc, u32Type, e32, shiftMask)
            .getResult();
    mlir::Value denormBits =
        rewriter
            .create<emitc::BitwiseLeftShiftOp>(loc, u32Type, denormBase,
                                               denormShift)
            .getResult();
    // normalized branch: (e - 1) << 23  (2^(e-128) normalized).
    mlir::Value oneU32 = rewriter.create<emitc::LiteralOp>(loc, u32Type, "1");
    mlir::Value eMinus1 =
        rewriter.create<emitc::SubOp>(loc, u32Type, e32, oneU32);
    mlir::Value shift23 = rewriter.create<emitc::LiteralOp>(loc, u32Type, "23");
    mlir::Value normBits =
        rewriter
            .create<emitc::BitwiseLeftShiftOp>(loc, u32Type, eMinus1, shift23)
            .getResult();
    mlir::Value bitsVal =
        rewriter
            .create<emitc::ConditionalOp>(loc, u32Type, isDenorm, denormBits,
                                          normBits)
            .getResult();
    // Materialize bits into an lvalue so its address can be taken for the pun.
    auto bitsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(u32Type), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(loc, bitsVar, bitsVal);
    // float scale_x = *(const float *)&bits;  -- the reinterpret pointer pun
    // (the structured analogue of the sanctioned fp16 pointer pun).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "e8m0_reinterpret_float"));
    mlir::Value bitsAddr =
        rewriter.create<emitc::ApplyOp>(loc, constU32PtrType, "&", bitsVar)
            .getResult();
    mlir::Value floatPtr =
        rewriter.create<emitc::CastOp>(loc, constFloatPtrType, bitsAddr)
            .getResult();
    mlir::Value floatIdx0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value floatElem =
        rewriter
            .create<emitc::SubscriptOp>(
                loc,
                mlir::cast<mlir::TypedValue<emitc::PointerType>>(floatPtr),
                floatIdx0)
            .getResult();
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Value scaleConst =
        rewriter.create<emitc::LoadOp>(loc, constFloatType, floatElem)
            .getResult();
    // Drop the const qualifier so the downstream fp32 fold multiplies two
    // plain `float` operands (emitc binary ops require matching element types).
    return rewriter.create<emitc::CastOp>(loc, floatType, scaleConst)
        .getResult();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMXFP4Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Thin shim: mxfp4 codebook_gather_nibble / half-block / SumiTimesScales
    // (E8M0 weight scale, dual weight/activation quant offsets) instance of the
    // descriptor-driven emitFlatBlockDot -- the FP4 codebook sibling of iq4_nl.
    // The gather + product + reduce + fold are byte-identical to iq4_nl; only the
    // weight-scale source (E8M0 vs fp16) and the quant offsets differ, both
    // descriptor fields. Enforce the codebook I7 fail-closed anchor guard, derive
    // the descriptor + the scheduled BlockDotFacts, emit the shared body.
    tcrvrvv::GgmlBlockDotMXFP4Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotMXFP4Q80Op>(op))
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

    // I7 FAIL-CLOSED (SAME as the iq4_nl sibling): an UN-scheduled (attr-less, no
    // integer_core_lmul) codebook op must NOT lower. The codebook gather indexes a
    // broadcast 16-entry table; the emitter's default m1 anchor has gather VLMAX < 16
    // below VLEN=128, so a high nibble index would silently read 0. The sub-128 pass
    // run leaves the op attr-less with NO minimum_vlen (the codebook class is
    // Zvl128b-gated, no legal anchor exists), verifier-indistinguishable from the
    // legal pre-schedule input -- but the lowering boundary CAN refuse it: every
    // VLEN>=128 path stamps a legal m1/mf2 first, so an anchor-less op here is the
    // unsafe sub-128 leftover. Refuse fail-closed.
    if (!blockDot.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          blockDot,
          "refusing to lower an UN-scheduled codebook op (no integer_core_lmul): "
          "the codebook class is Zvl128b-gated and the emitter's default m1 anchor "
          "cannot host the 16-entry gather below VLEN=128 (a nibble index >= VLMAX "
          "silently reads 0); the gearbox must stamp a legal anchor first "
          "(materialize-schedule on a VLEN>=128 target) -- fail-closed (I7)");

    std::optional<FlatBlockDotDescriptor> descriptor =
        deriveFlatBlockDotDescriptor(blockDot.getOperation());
    if (!descriptor)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot kind not flat-codebook");
    BlockDotFacts facts =
        deriveBlockDotFacts(blockDot, descriptor->defaultCoreLmul);
    return emitFlatBlockDot(rewriter, loc, weightBase, activationBase, output,
                            blockDot.getResult(), avlArg, sizeType, valueMap,
                            blockDot.getTCRVEmitCLowerableSourceOpName(),
                            blockDot.getTCRVEmitCLowerableSourceRole(), facts,
                            *descriptor);
  }

mlir::LogicalResult VariantToEmitCFunc::emitNVFP4Q8_0BlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotNVFP4Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotNVFP4Q80Op>(op))
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
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The codebook gather REQUIRES the m1 anchor (VLMAX >= 16); the verifier fixes
    // it to m1. The i8 source LMUL ("m1") drives the i16 product LMUL ("m2").
    llvm::StringRef coreLmul = "m1";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    WideningChain wideningChain = deriveWideningChain(coreLmul);
    llvm::StringRef wideLmul = wideningChain.l16;
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The super-block-format structural facts come straight off the typed attrs
    // (I4). NOTE the nvfp4-specific layout: four UE4M3 scales at +0..3, then the FP4
    // nibbles at the WEIGHT quant offset (+4); the q8 quants at the ACTIVATION quant
    // offset (+2, after the inline fp16 scale), the per-sub-block q8 high half at +8.
    int64_t qk = blockDot.getQk();           // 64 (QK_NVFP4)
    int64_t qkSub = blockDot.getQkSub();     // 16 (QK_NVFP4_SUB)
    int64_t weightStride = blockDot.getWeightBlockStride();       // 36
    int64_t activationStride = blockDot.getActivationBlockStride(); // 34
    int64_t weightQuantOffset = blockDot.getWeightQuantByteOffset(); // 4
    int64_t activationQuantOffset = blockDot.getActivationQuantByteOffset(); // 2
    int64_t highOffset = blockDot.getActivationHighByteOffset();  // 8
    int64_t numSubBlocks = qk / qkSub;       // 4
    int64_t subHalf = qkSub / 2;             // 8 lanes per strip
    int64_t weightSubStride = qkSub / 2;     // 8 nibble bytes per sub-block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The 16-entry FP4 (e2m1) codebook is a STRUCTURAL fact off the typed attr
    // (I4 mirror) -- REUSED from mxfp4 (the SAME kvalues_mxfp4 table). Emit it as a
    // `static const int8_t[16]` decl ONCE, then broadcast it into `values` via vle8
    // ONCE above the block loop (the gather table).
    llvm::ArrayRef<int8_t> codebook = blockDot.getCodebook();
    {
      std::string decl = "static const int8_t tcrv_nvfp4_kvalues[16] = {";
      for (size_t i = 0; i < codebook.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(codebook[i]));
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK;  (QK = 64: super-blocks, NOT q8 blocks)
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // vint8m1_t values = __riscv_vle8_v_i8m1(tcrv_nvfp4_kvalues, 16);  (the FP4
    // codebook table broadcast into a vreg ONCE; reused by every gather).
    std::string tableLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    mlir::Value values = emitOpaqueCallBuilt(
        rewriter, loc, i8CoreType, tableLoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value tableName = rewriter.create<emitc::LiteralOp>(
              loc, i8PtrType, "tcrv_nvfp4_kvalues");
          return {tableName, sizeLit(codebook.size())};
        },
        llvm::StringRef("codebook_table_load"));

    // The q8_0 fp16->fp32 scale read (the ONE sanctioned opaque scalar piece).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    // The STRUCTURED UE4M3 -> fp32 HALF weight scale (the genuinely-new NVFP4-class
    // piece): ggml_ue4m3_to_fp32(e) (ggml-impl.h). UE4M3 is UNSIGNED (4 exp bias-7,
    // 3 man, NO sign bit). e==0 || e==0x7F -> 0.0f (two specials); else
    // exp=(e>>3)&0xF, man=e&7, raw=(exp==0)?ldexpf(man,-9):ldexpf(1+man/8, exp-7),
    // result = raw * 0.5f (the HALF compensation for the doubled codebook). Every
    // UE4M3 scale lies in comfortably-normal fp32 (~[2^-10, 224]) -- no FTZ risk --
    // so ggml's EXACT ldexpf arithmetic is replicated structurally and is byte-
    // identical to _generic by construction (same ops, same libm, same board).
    // Emitted as structured emitc nodes (load e, two cmp specials, exp/man split,
    // the two ldexpf branches via call_opaque, the *0.5f, two conditional selects).
    auto emitUE4M3Scale = [&](mlir::Value scaleBytePtr) -> mlir::Value {
      // uint8_t e = *(const uint8_t *)(scaleBytePtr);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "ue4m3_scale_load"));
      mlir::Value xbU8 =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, scaleBytePtr)
              .getResult();
      mlir::Value eIdx0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value eElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, mlir::cast<mlir::TypedValue<emitc::PointerType>>(xbU8),
                  eIdx0)
              .getResult();
      mlir::Value e =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, eElem).getResult();
      // uint32_t e32 = (uint32_t) e;  (the bit ops run in the uint32_t domain).
      mlir::Value e32 =
          rewriter.create<emitc::CastOp>(loc, u32Type, e).getResult();

      // exp = (e >> 3) & 0xF;  man = e & 0x7;  -- the unsigned 4-exp / 3-man split.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "ue4m3_exp_man_split"));
      mlir::Value three = rewriter.create<emitc::LiteralOp>(loc, u32Type, "3");
      mlir::Value expMask = rewriter.create<emitc::LiteralOp>(loc, u32Type, "0xF");
      mlir::Value expShifted =
          rewriter.create<emitc::BitwiseRightShiftOp>(loc, u32Type, e32, three)
              .getResult();
      mlir::Value expU =
          rewriter.create<emitc::BitwiseAndOp>(loc, u32Type, expShifted, expMask)
              .getResult();
      mlir::Value manMask = rewriter.create<emitc::LiteralOp>(loc, u32Type, "0x7");
      mlir::Value manU =
          rewriter.create<emitc::BitwiseAndOp>(loc, u32Type, e32, manMask)
              .getResult();
      // int exp = (int) expU;  int man = (int) manU;  -- ldexpf/man arithmetic in int.
      mlir::Value expInt =
          rewriter.create<emitc::CastOp>(loc, intType, expU).getResult();
      mlir::Value manInt =
          rewriter.create<emitc::CastOp>(loc, intType, manU).getResult();
      mlir::Value manFloat =
          rewriter.create<emitc::CastOp>(loc, floatType, manInt).getResult();

      // The two ldexpf branches (ggml's exact arithmetic):
      //   denormal (exp==0): ldexpf((float)man, -9)
      //   normal:            ldexpf(1.0f + (float)man / 8.0f, exp - 7)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "ue4m3_ldexpf_branches"));
      // denormBranch = ldexpf(manFloat, -9)
      mlir::Value m9 =
          rewriter.create<emitc::LiteralOp>(loc, intType, "-9");
      mlir::Value denormRaw =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           "ldexpf",
                                           mlir::ValueRange{manFloat, m9})
              .getResult(0);
      // normMantissa = 1.0f + manFloat / 8.0f
      mlir::Value oneF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
      mlir::Value eightF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "8.0f");
      mlir::Value manDiv8 =
          rewriter.create<emitc::DivOp>(loc, floatType, manFloat, eightF);
      mlir::Value normMantissa =
          rewriter.create<emitc::AddOp>(loc, floatType, oneF, manDiv8);
      // normExp = exp - 7
      mlir::Value sevenI =
          rewriter.create<emitc::LiteralOp>(loc, intType, "7");
      mlir::Value normExp =
          rewriter.create<emitc::SubOp>(loc, intType, expInt, sevenI);
      mlir::Value normRaw =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{floatType}, "ldexpf",
                  mlir::ValueRange{normMantissa, normExp})
              .getResult(0);
      // raw = (exp == 0) ? denormRaw : normRaw;
      mlir::Value zeroU =
          rewriter.create<emitc::LiteralOp>(loc, u32Type, "0");
      mlir::Value isDenorm =
          rewriter
              .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::eq, expU,
                                    zeroU)
              .getResult();
      mlir::Value raw =
          rewriter
              .create<emitc::ConditionalOp>(loc, floatType, isDenorm, denormRaw,
                                            normRaw)
              .getResult();
      // scaled = raw * 0.5f  (the HALF compensation for the doubled codebook).
      mlir::Value halfF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.5f");
      mlir::Value scaled =
          rewriter.create<emitc::MulOp>(loc, floatType, raw, halfF);

      // The two specials: e == 0 || e == 0x7F -> 0.0f. Evaluated AFTER the
      // arithmetic and selected via two structured conditionals (both branch values
      // exist; the select discards the arithmetic when special). This matches
      // ggml's early `if (x == 0 || x == 0x7F) return 0.0f;` byte-exactly: when e is
      // 0 the exp==0 denorm branch yields ldexpf(0,-9)=0 anyway, but e==0x7F would
      // otherwise hit the NORMAL branch (exp=15,man=7 -> 240), so the special MUST be
      // applied explicitly.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "ue4m3_specials"));
      mlir::Value zeroF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
      mlir::Value zeroByte =
          rewriter.create<emitc::LiteralOp>(loc, u32Type, "0");
      mlir::Value specZero =
          rewriter.create<emitc::LiteralOp>(loc, u32Type, "0x7F");
      mlir::Value isZero =
          rewriter
              .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::eq, e32,
                                    zeroByte)
              .getResult();
      mlir::Value isSpec7F =
          rewriter
              .create<emitc::CmpOp>(loc, boolType, emitc::CmpPredicate::eq, e32,
                                    specZero)
              .getResult();
      mlir::Value isSpecial =
          rewriter
              .create<emitc::LogicalOrOp>(loc, boolType, isZero, isSpec7F)
              .getResult();
      return rewriter
          .create<emitc::ConditionalOp>(loc, floatType, isSpecial, zeroF, scaled)
          .getResult();
    };

    // The CODEBOOK decode + asymmetric product + reduce for ONE sub-block strip
    // (REUSED from the mxfp4 sibling, with the nvfp4 8-lane sub-block geometry): the
    // weight nibble byte loads UNSIGNED from xqs, the two q8 halves load SIGNED from
    // yqs and yqs + highOffset. Returns the scalar i32 sumi for this sub-block.
    auto emitSubBlockReduce = [&](mlir::Value xqs, mlir::Value yqs,
                                  mlir::Value vl) -> mlir::FailureOr<mlir::Value> {
      std::string wLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      mlir::Value w = emitOpaqueCall(rewriter, loc, u8CoreType, wLoadCallee,
                                     mlir::ValueRange{xqs, vl}, opName, role);
      std::string yLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, yLoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      mlir::Value yHigh = rewriter.create<emitc::AddOp>(
          loc, i8PtrType, yqs, sizeLit(highOffset));
      mlir::Value y0 = loadY(yqs);
      mlir::Value y1 = loadY(yHigh);

      // The codebook nibble decode: split into the two UNSIGNED index lanes, then
      // gather each through the broadcast table -> signed-i8 weight lanes v0/v1.
      auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                         llvm::StringRef amount) -> mlir::Value {
        std::string callee = ("__riscv_" + mnemonic + "_u8" + coreLmul).str();
        return emitOpaqueCallBuilt(
            rewriter, loc, u8CoreType, callee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value amt = rewriter.create<emitc::LiteralOp>(
                  loc, emitc::OpaqueType::get(ctx, "int"), amount.str());
              return {src, amt, vl};
            });
      };
      mlir::Value idxLow = u8ImmOp("vand_vx", w, "0x0F");
      mlir::Value idxHigh = u8ImmOp("vsrl_vx", w, "0x04");
      std::string gatherCallee = ("__riscv_vrgather_vv_i8" + coreLmul).str();
      auto gather = [&](mlir::Value idx) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, gatherCallee,
                              mlir::ValueRange{values, idx, vl}, opName, role);
      };
      mlir::Value v0 = gather(idxLow);
      mlir::Value v1 = gather(idxHigh);

      // The SAME asymmetric signed widening product the codebook siblings use.
      mlir::FailureOr<mlir::Value> product =
          emitOffsetBinaryProductFromDecodedValue(rewriter, loc, v0, v1, y0, y1,
                                                  vl, i16WideType, 16, wideLmul,
                                                  "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-sub-block scalar: seed lane0 = 0, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, seedCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumiSeed =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            mlir::Value one = sizeLit(1);
            return {sumiSeed, one};
          });
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red = emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                                       mlir::ValueRange{*product, seed, vl},
                                       opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
    };

    // The per-sub-block fp32 accumulate `sumf = sumf + dy * d * (float)sumi`
    // (ggml's nvfp4 order: dy * d, then * the integer sum; _generic is `dy * d *
    // (sumi_lo + sumi_hi)`). Grouped into ONE emitc.expression so mlir-translate
    // renders ONE C statement (the SAME FMA ggml fuses under -ffp-contract=on).
    // Caller invokes in STRICT ascending (ib, s_idx) order, preserving fp non-
    // associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumi, mlir::Value dY, mlir::Value scaleX) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumi).getResult();
        // dy * d FIRST (ggml nvfp4 scales-first order: `dy * d * (...)`).
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dY, scaleX);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scaleProduct,
                                          sumiFloat);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // Per-sub-block address arithmetic helper: base + (idxAdd)*stride + fixed.
    auto offsetPtr = [&](mlir::Value base, mlir::Type ptrType,
                         mlir::Value scaledIdx, int64_t stride,
                         int64_t fixed) -> mlir::Value {
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, scaledIdx, sizeLit(stride));
      mlir::Value p = rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
      if (fixed != 0)
        p = rewriter.create<emitc::AddOp>(loc, ptrType, p, sizeLit(fixed));
      return p;
    };

    // The set-vl for the 8-lane sub-block strip (anchored at m1: VLMAX caps at the
    // table's 16 entries; here vl=8, so the gather's table register still holds all
    // 16 entries at VLEN>=128). Computed ONCE per super-block iteration (constant 8).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");

    // The outer super-block loop: for (ib = 0; ib < nb; ++ib).
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    mlir::LogicalResult status = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // const uint8_t *xb = vx + ib*36;  (super-block weight base)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "super_block_base_x"));
      mlir::Value ibStride =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(weightStride));
      mlir::Value xb =
          rewriter.create<emitc::AddOp>(loc, weightPtrType, weightBase, ibStride);

      // size_t q8base = 2*ib;  (the FIRST of the two q8_0 blocks this super-block
      // consumes; sub-block s reads q8 block q8base + s/2).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "q8_block_base_index"));
      mlir::Value q8Base =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(2));

      // The four sub-blocks, UNROLLED in STRICT ascending s order so the per-
      // sub-block fp32 fold preserves ggml's exact accumulation order.
      for (int64_t s = 0; s < numSubBlocks; ++s) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block"));
        // const uint8_t *scalePtr = xb + s;  (the s-th UE4M3 scale byte at +s).
        mlir::Value scalePtr =
            s == 0 ? xb
                   : rewriter
                         .create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(s))
                         .getResult();
        mlir::Value scaleX = emitUE4M3Scale(scalePtr);

        // const uint8_t *xqs = xb + 4 + s*8;  (the FP4 nibbles for sub-block s,
        // after the four scale bytes; weightQuantOffset == 4, sub-stride 8).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_weight_quants"));
        mlir::Value xqsFull = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, xb,
            sizeLit(weightQuantOffset + s * weightSubStride));
        mlir::Value xqs =
            rewriter.create<emitc::CastOp>(loc, u8PtrType, xqsFull).getResult();

        // const uint8_t *yb = vy + (2*ib + s/2)*34;  (the q8_0 block for this
        // sub-block; s/2 is the integer-division block selector within the pair).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_q8_base"));
        mlir::Value q8Idx =
            (s / 2) == 0
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, sizeType, q8Base,
                                            sizeLit(s / 2))
                      .getResult();
        mlir::Value yb = offsetPtr(activationBase, activationPtrType, q8Idx,
                                   activationStride, 0);

        // float dy = (float)*(const _Float16 *)yb;  (the q8_0 fp16 scale).
        mlir::Value dY = fp16Read(yb);

        // const int8_t *yqs = (const int8_t *)(yb + 2 + (s%2)*16);  (the q8 quant
        // half this sub-block pairs with: low half of the q8 block for even s,
        // high half for odd s).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_q8_quants"));
        int64_t q8FixedOffset = activationQuantOffset + (s % 2) * qkSub;
        mlir::Value yqsFull = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, yb, sizeLit(q8FixedOffset));
        mlir::Value yqs =
            rewriter.create<emitc::CastOp>(loc, i8PtrType, yqsFull).getResult();

        // The 8-lane strip setvl + the codebook integer core -> sumi.
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, setvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(subHalf)};
            });
        mlir::FailureOr<mlir::Value> sumi =
            emitSubBlockReduce(xqs, yqs, vl);
        if (mlir::failed(sumi)) {
          status = mlir::failure();
          break;
        }
        emitFold(*sumi, dY, scaleX);
      }
    }
    if (mlir::failed(status))
      return mlir::failure();

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

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
