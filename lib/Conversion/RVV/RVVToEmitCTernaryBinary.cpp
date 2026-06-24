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

// VariantToEmitCFunc ternary + binary emit methods: iq1_s / iq1_m and
// tq1_0 / tq2_0. Split out of RVVToEmitC.cpp as a pure code move; the emitted C
// is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitIQ1SQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ1SQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ1SQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq1_s block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq1_s block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t subBlock = blockDot.getSubBlock();                      //  32
    int64_t weightStride = blockDot.getWeightBlockStride();         //  50
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   2
    int64_t qhOffset = blockDot.getWeightQhByteOffset();            //  34
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t bsumsOffset = blockDot.getActivationBsumsByteOffset();  // 260
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t groupsPerSub = 4;  // 4 grid groups per sub-block (l=0..3)

    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    // vluxei16 IQ-gather revectorization (emitter-maturity, adopts ggml's
    // __riscv_vluxei16 indexed grid gather). One 32-lane body per sub-block:
    //   vluxei16_v_i64m2 (gather 4 grid u64 entries via u16 byte-offset indices)
    //   -> reinterpret i8m2 (32 ternary int8 grid values) -> vwmul i16m4 -> ONE
    //   vwredsum per sub-block. The 4 scalar 11-bit index computations are KEPT
    //   byte-exact; only the order-free integer gather+product+reduce is fused.
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16mf2Type = emitc::OpaqueType::get(ctx, "vuint16mf2_t");
    mlir::Type i64m2Type = emitc::OpaqueType::get(ctx, "vint64m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i16m4Type = emitc::OpaqueType::get(ctx, "vint16m4_t");
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));

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
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type i16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int16_t"));
    mlir::Type constU16Type = emitc::OpaqueType::get(ctx, "const uint16_t");
    mlir::Type constI16Type = emitc::OpaqueType::get(ctx, "const int16_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The TERNARY GRID codebook is a STRUCTURAL fact off the typed attr (I4 mirror):
    // the 2048 uint64 iq1s_grid literals, emitted ONCE as a `static const
    // int64_t[2048]` decl (rendering ggml's exact hex literals) and read as bytes
    // through a (const int8_t *) cast -- copying ggml's source method (`grid = (const
    // int8_t *)(iq1s_grid + idx)`). The grid bytes are {0x00,0x01,0xff} = {0,+1,-1};
    // reading 0xff as int8 yields -1, byte-identical to ggml's int8 grid read.
    // The grid is rendered as a `static const uint64_t[2048]` (matching ggml's own
    // `uint64_t iq1s_grid` type): the ternary entries have the high bit set (e.g.
    // 0xffffffffffffffff = all -1), which would not narrow into a signed int64_t under
    // C++11. Reading it through a `(const int8_t *)` byte cast still yields the signed
    // ternary value (0xff -> -1), byte-identical to ggml's grid read.
    llvm::ArrayRef<int64_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const uint64_t tcrv_iq1s_grid[2048] = {";
      for (size_t i = 0; i < grid.size(); ++i) {
        if (i)
          decl += ", ";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%016llxULL",
                      static_cast<unsigned long long>(
                          static_cast<uint64_t>(grid[i])));
        decl += buf;
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // float sumf = 0.0f;  (function-scoped accumulator across the super-block loop)
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

    // const uint64_t *grid = tcrv_iq1s_grid;  (the u64 grid base; the vluxei16 gather
    // reads it as (const int64_t *) with u16 byte-offset indices idx*8 -- exactly
    // ggml's (const int64_t *)iq1s_grid indexed gather.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type u64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u64PtrType, "tcrv_iq1s_grid");

    // Per-super-block base address arithmetic: xb = vx + ibl*50; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (the single qs index byte; the shift amount stays positive so no
    // sign-extension hazard).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, emitc::OpaqueType::get(ctx, "const uint8_t"), elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
    };

    // int x = (int)qh[i];  -- a structured uint16 load from a `const uint16_t *` then
    // a cast to int (the per-sub-block qh word carrying the grid-high-3-bit fields,
    // the 3-bit scale @12..14, the delta sign @15; all 16 bits are consumed).
    auto loadU16AsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u16 =
          rewriter.create<emitc::LoadOp>(loc, constU16Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u16).getResult();
    };

    // int x = (int)bsums[i];  -- a structured int16 load from a `const int16_t *` then
    // a cast to int (the q8_K per-16-element sums folded by the delta term; signed).
    auto loadI16AsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value i16 =
          rewriter.create<emitc::LoadOp>(loc, constI16Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, i16).getResult();
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

      // d = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);  (ONCE
      // per super-block; the fp16 weight scale times the fp32 q8_K scale).
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dxAddr})
              .getResult(0);
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
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // const uint8_t *qs = xb + 2;  const uint16_t *qh = xb + 34;
      // const int8_t *q8 = yb + 4;   const int16_t *bsums = yb + 260;
      mlir::Value qsBase0 = xb;
      if (qsOffset != 0)
        qsBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qsOffset));
      mlir::Value qsBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBase0).getResult();
      mlir::Value qhBase0 = xb;
      if (qhOffset != 0)
        qhBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qhOffset));
      mlir::Value qhBase =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, qhBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();
      mlir::Value bsumsBase0 = yb;
      if (bsumsOffset != 0)
        bsumsBase0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                   sizeLit(bsumsOffset));
      mlir::Value bsumsBase =
          rewriter.create<emitc::CastOp>(loc, i16PtrType, bsumsBase0).getResult();

      // int32_t sumi = 0;  int32_t sumi1 = 0;  (the TWO integer super-block
      // accumulators -- the grid dot and the delta-bsum term -- reset per super-block
      // and kept SEPARATE across all 8 sub-blocks; ggml's per-super-block sumi/sumi1).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi1", opName, role));
      auto sumi1Var = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumi1Var, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib = 0..7), fully unrolled. Each sub-block has 4
      // single-byte qs index bytes (qs[ib*4 + l]), one uint16 qh word (qh[ib]) and two
      // int16 bsums (bsums[2*ib+0], bsums[2*ib+1]); the q8 cursor advances 8 per group.
      for (int64_t ib = 0; ib < numSubBlocks; ++ib) {
        // DELTA: int qhw = qh[ib];  ls = 2*((qhw>>12)&7)+1;  delta = 1 - 2*((qhw>>15)&1).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_word_scale_delta"));
        mlir::Value qhWord = loadU16AsInt(qhBase, ib);

        // ls = 2*((qhw >> 12) & 7) + 1.
        mlir::Value scaleField =
            rewriter
                .create<emitc::BitwiseAndOp>(
                    loc, intType,
                    rewriter
                        .create<emitc::BitwiseRightShiftOp>(loc, intType, qhWord,
                                                            intLit(12))
                        .getResult(),
                    intLit(7))
                .getResult();
        mlir::Value ls =
            rewriter
                .create<emitc::AddOp>(
                    loc, intType,
                    rewriter
                        .create<emitc::MulOp>(loc, intType, scaleField,
                                              intLit(2))
                        .getResult(),
                    intLit(1))
                .getResult();

        // delta = 1 - 2*((qhw >> 15) & 1)   (= +1 if bit15==0, -1 if bit15==1).
        mlir::Value signBit =
            rewriter
                .create<emitc::BitwiseAndOp>(
                    loc, intType,
                    rewriter
                        .create<emitc::BitwiseRightShiftOp>(loc, intType, qhWord,
                                                            intLit(15))
                        .getResult(),
                    intLit(1))
                .getResult();
        mlir::Value delta =
            rewriter
                .create<emitc::SubOp>(
                    loc, intType, intLit(1),
                    rewriter
                        .create<emitc::MulOp>(loc, intType, signBit, intLit(2))
                        .getResult())
                .getResult();

        // The continuous q8 cursor for this sub-block's 4 groups.
        mlir::Value q8Group =
            (ib == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib * subBlock))
                      .getResult();

        // int32_t lacc = 0;  (the i32m1 reduction seed for the sub-block's ternary
        // grid dot; integer add is order-free).
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, seedCallee));
        mlir::Value zeroSeed =
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
        mlir::Value laccAcc =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             seedCallee,
                                             mlir::ValueRange{zeroSeed,
                                                              sizeLit(1)})
                .getResult(0);

        // uint16_t tmp[4];  (scratch holding the 4 grid byte-offset indices idx*8;
        // EEW=16 byte offsets feed the vluxei16 indexed gather. Max idx 2047 -> 2047
        // * 8 = 16376 < 65535 fits u16.)
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("idxoff", opName, role));
        mlir::Type idxArrayType =
            emitc::ArrayType::get({groupsPerSub}, u16ElemType);
        auto idxVar = rewriter.create<emitc::VariableOp>(
            loc, idxArrayType, emitc::OpaqueAttr::get(ctx, ""));
        auto idxArray =
            llvm::cast<mlir::TypedValue<emitc::ArrayType>>(idxVar.getResult());

        // KEEP the 4 scalar 11-bit index computations (byte-exact, untouched), each
        // shifted <<3 to a byte offset and stored into tmp[l].
        for (int64_t l = 0; l < groupsPerSub; ++l) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "ternary_grid_index"));
          // idx = qs[ib*4 + l] | (((qhw >> (3*l)) & 7) << 8).  The low 8 bits are a
          // single qs index byte; the high 3 bits are taken from the (3*l)-shifted qh
          // field masked to bits [8,10] (shift 0,3,6,9 for l=0,1,2,3). The whole index
          // stays within [0,2047].
          mlir::Value qsIdxByte =
              loadByteAsInt(qsBase, ib * groupsPerSub + l);
          mlir::Value qhField =
              rewriter
                  .create<emitc::BitwiseAndOp>(
                      loc, intType,
                      rewriter
                          .create<emitc::BitwiseRightShiftOp>(loc, intType,
                                                              qhWord,
                                                              intLit(3 * l))
                          .getResult(),
                      intLit(7))
                  .getResult();
          mlir::Value qhHighBits =
              rewriter
                  .create<emitc::BitwiseLeftShiftOp>(loc, intType, qhField,
                                                     intLit(8))
                  .getResult();
          mlir::Value idx =
              rewriter
                  .create<emitc::BitwiseOrOp>(loc, intType, qsIdxByte,
                                              qhHighBits)
                  .getResult();
          // tmp[l] = (uint16_t)(idx << 3);  (idx*8 byte offset into the u64 grid).
          mlir::Value byteOff =
              rewriter
                  .create<emitc::BitwiseLeftShiftOp>(loc, intType, idx,
                                                     intLit(3))
                  .getResult();
          mlir::Value byteOffU16 =
              rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff)
                  .getResult();
          mlir::Value tmpIdx = rewriter.create<emitc::LiteralOp>(
              loc, rewriter.getIndexType(), std::to_string(l));
          mlir::Value tmpElem =
              rewriter
                  .create<emitc::SubscriptOp>(loc, idxArray,
                                              mlir::ValueRange{tmpIdx})
                  .getResult();
          rewriter.create<emitc::AssignOp>(
              loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(tmpElem),
              byteOffU16);
        }

        // vuint16mf2_t vidx = __riscv_vle16_v_u16mf2(&tmp[0], 4);  (index EMUL =
        // (EEW_idx/SEW_data)*LMUL_data = (16/64)*2 = 1/2 -> mf2 for the i64m2 gather.)
        mlir::Value idxBaseIndex0 =
            rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
        mlir::Value idxBaseElem0 =
            rewriter
                .create<emitc::SubscriptOp>(loc, idxArray,
                                            mlir::ValueRange{idxBaseIndex0})
                .getResult();
        mlir::Value idxBase =
            rewriter
                .create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", idxBaseElem0)
                .getResult();
        std::string idxLoadCallee =
            riscvIntrinsicName("vle", 16, "mf2", "u16");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, idxLoadCallee));
        mlir::Value vidx =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{u16mf2Type}, idxLoadCallee,
                    mlir::ValueRange{idxBase, sizeLit(groupsPerSub)})
                .getResult(0);

        // grid64 = (const int64_t *)tcrv_iq1s_grid;  (the u64 grid as int64 base for
        // the indexed gather -- exactly ggml's (const int64_t *)iq1s_grid.)
        mlir::Value grid64 =
            rewriter.create<emitc::CastOp>(loc, i64PtrType, gridArrayName)
                .getResult();

        // vint64m2_t g64 = __riscv_vluxei16_v_i64m2(grid64, vidx, 4);  -- the HARDWARE
        // indexed gather of the 4 grid u64 entries (ggml's __riscv_vluxei16), then
        // reinterpret to i8m2 = 32 signed ternary grid bytes.
        std::string gatherCallee =
            riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", "m2");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, gatherCallee));
        mlir::Value gathered =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{i64m2Type}, gatherCallee,
                    mlir::ValueRange{grid64, vidx, sizeLit(groupsPerSub)})
                .getResult(0);
        std::string reinterpretCallee = "__riscv_vreinterpret_v_i64m2_i8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reinterpretCallee));
        mlir::Value gridV =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8m2Type},
                                             reinterpretCallee,
                                             mlir::ValueRange{gathered})
                .getResult(0);

        // vint8m2_t q8v = __riscv_vle8_v_i8m2(q8Group, 32);  (the full 32-lane
        // sub-block activations, one contiguous load.)
        std::string q8LoadCallee = riscvIntrinsicName("vle", 8, "m2", "i8");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, q8LoadCallee));
        mlir::Value q8V =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{i8m2Type}, q8LoadCallee,
                    mlir::ValueRange{q8Group, sizeLit(subBlock)})
                .getResult(0);

        // p = __riscv_vwmul_vv_i16m4(grid, q8v, 32);  (signed widening product; the
        // ternary grid bytes are ALREADY signed {-1,0,+1}, NO sign-apply.)
        std::string wmulCallee = "__riscv_vwmul_vv_i16m4";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, wmulCallee));
        mlir::Value product =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{i16m4Type}, wmulCallee,
                    mlir::ValueRange{gridV, q8V, sizeLit(subBlock)})
                .getResult(0);

        // lacc = __riscv_vwredsum_vs_i16m4_i32m1(p, lacc, 32);  -- ONE reduction per
        // sub-block (per the byte-exact guard: ls is per-sub-block). The 32-lane sum
        // over |product|<=127*32=4064 fits i32 and is integer-associative with the
        // prior 4x8-lane reductions -> byte-exact.
        std::string reduceCallee = "__riscv_vwredsum_vs_i16m4_i32m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reduceCallee));
        laccAcc =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                    mlir::ValueRange{product, laccAcc, sizeLit(subBlock)})
                .getResult(0);

        // int lsum = __riscv_vmv_x_s_i32m1_i32(laccAcc);  (extract the ternary dot of
        // this sub-block's 4 groups).
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, extractCallee));
        mlir::Value lsum =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                             extractCallee,
                                             mlir::ValueRange{laccAcc})
                .getResult(0);

        // sumi = sumi + ls * lsum;  (integer accumulation; order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sumi_accumulate"));
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, ls).getResult();
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
        mlir::Value lsLsum =
            rewriter.create<emitc::MulOp>(loc, i32Type, lsI32, lsum).getResult();
        mlir::Value sumiNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumiCur, lsLsum)
                .getResult();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);

        // DELTA term: sumi1 = sumi1 + ls * delta * (bsums[2*ib+0] + bsums[2*ib+1]);
        // (the NEW mechanism -- a per-sub-block +-ls constant times the q8 16-element
        // sums folded in the integer domain).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "delta_bsum_accumulate"));
        mlir::Value bsum0 = loadI16AsInt(bsumsBase, 2 * ib + 0);
        mlir::Value bsum1 = loadI16AsInt(bsumsBase, 2 * ib + 1);
        mlir::Value bsumPair =
            rewriter.create<emitc::AddOp>(loc, intType, bsum0, bsum1)
                .getResult();
        mlir::Value lsDelta =
            rewriter.create<emitc::MulOp>(loc, intType, ls, delta).getResult();
        mlir::Value deltaTerm =
            rewriter.create<emitc::MulOp>(loc, intType, lsDelta, bsumPair)
                .getResult();
        mlir::Value deltaTermI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, deltaTerm).getResult();
        mlir::Value sumi1Cur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumi1Var).getResult();
        mlir::Value sumi1Next =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumi1Cur, deltaTermI32)
                .getResult();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi1", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumi1Var, sumi1Next);
      }

      // sumf = sumf + d * ((float)sumi + 0.125f * (float)sumi1);  -- ONE
      // emitc.expression so it renders as ggml's single C statement and the compiler
      // fuses the SAME contraction under -ffp-contract=on/default/fast. The 0.125
      // (IQ1S_DELTA) is applied EXACTLY ONCE here, to (float)sumi1 only; the inner add
      // stays in the expression tree. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate_with_delta"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumi1Final =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumi1Var).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      // The 0.125f (IQ1S_DELTA) literal is materialized OUTSIDE the expression region
      // (an emitc.literal is not a permitted expression-body op); it feeds the
      // expression as an operand exactly like d/sumi/sumi1.
      mlir::Value deltaConst =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f");
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal).getResult();
        mlir::Value sumi1Float =
            rewriter.create<emitc::CastOp>(loc, floatType, sumi1Final)
                .getResult();
        mlir::Value deltaScaled =
            rewriter.create<emitc::MulOp>(loc, floatType, deltaConst, sumi1Float)
                .getResult();
        mlir::Value inner =
            rewriter.create<emitc::AddOp>(loc, floatType, sumiFloat, deltaScaled)
                .getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, inner).getResult();
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm)
                .getResult();
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    }

    // *s = sumf;  (iq1_s applies NO trailing factor -- the store is the bare
    // accumulator; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq1_s block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitIQ1MQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ1MQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ1MQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq1_m block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq1_m block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u16Type = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t subBlock = blockDot.getSubBlock();                      //  32
    int64_t weightStride = blockDot.getWeightBlockStride();         //  56
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   0
    int64_t qhOffset = blockDot.getWeightQhByteOffset();            //  32
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();    //  48
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t groupsPerSub = 4;  // 4 grid groups per sub-block (l=0..3)
    int64_t groupLanes = 8;    // 8 grid values per group

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // vluxei16 IQ-gather revectorization (emitter-maturity, adopts ggml's
    // __riscv_vluxei16 indexed grid gather -- same grid/index-width as iq1_s).
    // iq1_m's grid dot must split into TWO halves (groups {0,1} -> sum1[0],
    // groups {2,3} -> sum1[1]) because the halves carry DIFFERENT scales
    // (ls1 vs ls2). So per HALF (h=0,1) we gather 2 grid u64 entries via a
    // vluxei16_v_i64m1 (index EMUL = (16/64)*1 = 1/4 -> mf4), reinterpret to
    // i8m1 (16 ternary int8 grid values), vwmul i16m2 over the 16 lanes, then
    // ONE vwredsum per half into sum1[h]. This is byte-exact: the 16-lane
    // half reduce = Sigma(group 2h) + Sigma(group 2h+1), integer-identical to
    // the original two 8-lane reduces summed into sum1[h]. Shape-agnostic (a
    // half is ALWAYS 2 u64 = 16 bytes = i64m1, independent of VLEN), so no
    // VLEN128-only lane assumption. The 4 scalar 11-bit index computations are
    // KEPT byte-exact; only the order-free integer gather+product+reduce of the
    // grid dot is fused. The sum2/delta path is grid-INDEPENDENT and stays
    // per-group UNCHANGED.
    int64_t halvesPerSub = 2;     // 2 halves (h=0,1) per sub-block
    int64_t groupsPerHalf = 2;    // 2 grid groups per half
    int64_t halfLanes = 16;       // 16 ternary grid bytes per half
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16mf4Type = emitc::OpaqueType::get(ctx, "vuint16mf4_t");
    mlir::Type i64m1Type = emitc::OpaqueType::get(ctx, "vint64m1_t");
    mlir::Type i8HalfType = i8CoreType;     // vint8m1_t (16 grid bytes per half)
    mlir::Type i16HalfType = i16WideType;   // vint16m2_t (16-lane products)
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));

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
    mlir::Type u16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint16_t"));
    mlir::Type u16LValuePtrType = emitc::PointerType::get(u16Type);
    mlir::Type constU16Type = emitc::OpaqueType::get(ctx, "const uint16_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The TERNARY GRID codebook is a STRUCTURAL fact off the typed attr (I4 mirror):
    // the SAME 2048 uint64 iq1s_grid literals as iq1_s, emitted ONCE as a `static
    // const uint64_t[2048]` decl and read as bytes through a (const int8_t *) cast
    // (`grid = (const int8_t *)(iq1s_grid + idx)`). The grid bytes are {0,+1,-1};
    // reading 0xff as int8 yields -1, byte-identical to ggml's grid read.
    llvm::ArrayRef<int64_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const uint64_t tcrv_iq1m_grid[2048] = {";
      for (size_t i = 0; i < grid.size(); ++i) {
        if (i)
          decl += ", ";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%016llxULL",
                      static_cast<unsigned long long>(
                          static_cast<uint64_t>(grid[i])));
        decl += buf;
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // float sumf = 0.0f;  (function-scoped accumulator across the super-block loop)
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

    // const uint64_t *tcrv_iq1m_grid;  (the u64 grid table name; cast to a
    // (const int64_t *) base for the vluxei16 indexed gather inside the loop --
    // exactly ggml's (const int64_t *)iq1s_grid.)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_base"));
    mlir::Type u64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u64PtrType, "tcrv_iq1m_grid");

    // Per-super-block base address arithmetic: xb = vx + ibl*56; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (a single qs index byte or a uint8 qh byte; positive so no
    // sign-extension hazard).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter
              .create<emitc::LoadOp>(
                  loc, emitc::OpaqueType::get(ctx, "const uint8_t"), elem)
              .getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
    };

    // int x = (int)sc[i];  -- a structured uint16 load from a `const uint16_t *` then
    // a cast to int (one of the 4 packed scales[] words; carries the fp16 d
    // high-nibble AND the per-sub-block 3-bit scales low-bits; all positive).
    auto loadU16AsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u16 =
          rewriter.create<emitc::LoadOp>(loc, constU16Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u16).getResult();
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

      // const uint8_t *qs = xb + 0;  const uint8_t *qh = xb + 32;
      // const uint16_t *sc = (const uint16_t *)(xb + 48);  const int8_t *q8 = yb + 4;
      mlir::Value qsBase0 = xb;
      if (qsOffset != 0)
        qsBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qsOffset));
      mlir::Value qsBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBase0).getResult();
      mlir::Value qhBase0 = xb;
      if (qhOffset != 0)
        qhBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qhOffset));
      mlir::Value qhBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qhBase0).getResult();
      mlir::Value scBase0 = xb;
      if (scalesOffset != 0)
        scBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(scalesOffset));
      mlir::Value scBase =
          rewriter.create<emitc::CastOp>(loc, u16PtrType, scBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();

      // (a) RECONSTRUCT the packed iq1m_scale fp16 super-block scale (the NEW piece).
      // scbits = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000)
      // then READ the 16 bits AS _Float16 (a bit reinterpret -- ggml's union -- NOT a
      // numeric conversion). The reassembled bits are written into a uint16_t lvalue
      // and read via the address-of so the `(const _Float16 *)` cast is byte-identical.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "iq1m_scale_reconstruct"));
      mlir::Value sc0 = loadU16AsInt(scBase, 0);
      mlir::Value sc1 = loadU16AsInt(scBase, 1);
      mlir::Value sc2 = loadU16AsInt(scBase, 2);
      mlir::Value sc3 = loadU16AsInt(scBase, 3);
      // (sc[0] >> 12)
      mlir::Value t0 =
          rewriter
              .create<emitc::BitwiseRightShiftOp>(loc, intType, sc0, intLit(12))
              .getResult();
      // ((sc[1] >> 8) & 0x00f0)
      mlir::Value t1 =
          rewriter
              .create<emitc::BitwiseAndOp>(
                  loc, intType,
                  rewriter
                      .create<emitc::BitwiseRightShiftOp>(loc, intType, sc1,
                                                          intLit(8))
                      .getResult(),
                  intLit(0xf0))
              .getResult();
      // ((sc[2] >> 4) & 0x0f00)
      mlir::Value t2 =
          rewriter
              .create<emitc::BitwiseAndOp>(
                  loc, intType,
                  rewriter
                      .create<emitc::BitwiseRightShiftOp>(loc, intType, sc2,
                                                          intLit(4))
                      .getResult(),
                  intLit(0xf00))
              .getResult();
      // (sc[3] & 0xf000)
      mlir::Value t3 =
          rewriter
              .create<emitc::BitwiseAndOp>(loc, intType, sc3, intLit(0xf000))
              .getResult();
      mlir::Value scbitsInt =
          rewriter
              .create<emitc::BitwiseOrOp>(
                  loc, intType,
                  rewriter
                      .create<emitc::BitwiseOrOp>(
                          loc, intType,
                          rewriter
                              .create<emitc::BitwiseOrOp>(loc, intType, t0, t1)
                              .getResult(),
                          t2)
                      .getResult(),
                  t3)
              .getResult();
      // uint16_t scbits = (uint16_t)(...);  -- a uint16 lvalue holding the fp16 bits.
      auto scbitsVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(u16Type), emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value scbitsU16 =
          rewriter.create<emitc::CastOp>(loc, u16Type, scbitsInt).getResult();
      rewriter.create<emitc::AssignOp>(loc, scbitsVar, scbitsU16);
      // &scbits  -> read (float)*(const _Float16 *)(&scbits).
      mlir::Value scbitsAddr =
          rewriter
              .create<emitc::ApplyOp>(loc, u16LValuePtrType,
                                      rewriter.getStringAttr("&"), scbitsVar)
              .getResult();
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{scbitsAddr})
              .getResult(0);

      // d = dx * *(const float *)(yb + 0);  (the reconstructed fp16 weight scale times
      // the fp32 q8_K scale, ONE mul).
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
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // int32_t sumi1 = 0;  int32_t sumi2 = 0;  (the TWO integer super-block
      // accumulators -- the grid dot and the per-group delta term -- reset per
      // super-block and kept SEPARATE across all 8 sub-blocks; ggml's sumi1/sumi2).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi1", opName, role));
      auto sumi1Var = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumi1Var, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi2", opName, role));
      auto sumi2Var = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumi2Var, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib = 0..7), fully unrolled. Each sub-block has 4
      // single-byte qs index bytes (qs[ib*4 + l]), TWO uint8 qh bytes (qh[2*ib+0],
      // qh[2*ib+1]) carrying the index-high fields + delta signs, and one packed
      // scales word sc[ib/2]; the q8 cursor advances 8 per group.
      for (int64_t ib = 0; ib < numSubBlocks; ++ib) {
        // int qh0 = qh[2*ib + 0];  int qh1 = qh[2*ib + 1].
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_bytes_delta"));
        mlir::Value qh0 = loadByteAsInt(qhBase, 2 * ib + 0);
        mlir::Value qh1 = loadByteAsInt(qhBase, 2 * ib + 1);

        // delta[l] = 1 - 2*((qh[l/2] >> (l%2 ? 7 : 3)) & 1)  (= +1/-1). For l=0,1 use
        // qh0 (bits 3,7); for l=2,3 use qh1 (bits 3,7). Mirrors ggml's
        // delta[0]=qh0&8, delta[1]=qh0&0x80, delta[2]=qh1&8, delta[3]=qh1&0x80.
        auto deltaSign = [&](mlir::Value qhByte, int64_t bit) -> mlir::Value {
          mlir::Value b =
              rewriter
                  .create<emitc::BitwiseAndOp>(
                      loc, intType,
                      rewriter
                          .create<emitc::BitwiseRightShiftOp>(loc, intType,
                                                              qhByte, intLit(bit))
                          .getResult(),
                      intLit(1))
                  .getResult();
          return rewriter
              .create<emitc::SubOp>(
                  loc, intType, intLit(1),
                  rewriter.create<emitc::MulOp>(loc, intType, b, intLit(2))
                      .getResult())
              .getResult();
        };
        mlir::Value delta[4] = {deltaSign(qh0, 3), deltaSign(qh0, 7),
                                deltaSign(qh1, 3), deltaSign(qh1, 7)};

        // The continuous q8 cursor for this sub-block's 4 groups.
        mlir::Value q8Group =
            (ib == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib * subBlock))
                      .getResult();

        // The TWO per-half integer accumulators sum1[h] (grid dot) and sum2[h]
        // (delta term), h = 0 (groups 0..1), h = 1 (groups 2..3). They are plain int
        // lvalues holding scalar reductions (the generic's sum1[2]/sum2[2]).
        auto makeIntVar = [&](const char *name) {
          rewriter.create<emitc::VerbatimOp>(
              loc, localVariableComment(name, opName, role));
          auto v = rewriter.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(i32Type),
              emitc::OpaqueAttr::get(ctx, ""));
          rewriter.create<emitc::AssignOp>(
              loc, v, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));
          return v;
        };
        emitc::VariableOp sum1Var[2] = {makeIntVar("sum1_0"),
                                        makeIntVar("sum1_1")};
        emitc::VariableOp sum2Var[2] = {makeIntVar("sum2_0"),
                                        makeIntVar("sum2_1")};

        // uint16_t idxoff[4];  -- scratch holding the 4 grid byte offsets idx*8
        // (EEW=16 byte offsets feed the vluxei16 indexed gather; max idx 2047 ->
        // 2047*8 = 16376 < 65535 fits u16). The 4 scalar 11-bit index
        // computations are KEPT byte-exact (unchanged); only the destination
        // changes from a `grid_i8 + idx*8` pointer to a stored byte offset.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("idxoff", opName, role));
        mlir::Type idxArrayType =
            emitc::ArrayType::get({groupsPerSub}, u16ElemType);
        auto idxVar = rewriter.create<emitc::VariableOp>(
            loc, idxArrayType, emitc::OpaqueAttr::get(ctx, ""));
        auto idxArray =
            llvm::cast<mlir::TypedValue<emitc::ArrayType>>(idxVar.getResult());

        // The per-group index + delta loop (l=0..3): compute the byte offset
        // idxoff[l] and the grid-INDEPENDENT sum2/delta term EXACTLY as before
        // (the delta path is untouched -- it does NOT consume the grid gather).
        for (int64_t l = 0; l < groupsPerSub; ++l) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "ternary_grid_index"));
          // idx = qs[ib*4 + l] | (((qh[l/2]) << (8 - 4*(l%2))) & 0x700). The index-high
          // 3 bits come from bits 0..2 of qh[l/2] for even l (shift 8) and bits 4..6
          // for odd l (shift 4). qhSel selects qh0 (l<2) or qh1 (l>=2).
          mlir::Value qhSel = (l < 2) ? qh0 : qh1;
          int64_t sh = (l % 2) ? 4 : 8;
          mlir::Value qsIdxByte =
              loadByteAsInt(qsBase, ib * groupsPerSub + l);
          mlir::Value qhHighBits =
              rewriter
                  .create<emitc::BitwiseAndOp>(
                      loc, intType,
                      rewriter
                          .create<emitc::BitwiseLeftShiftOp>(loc, intType, qhSel,
                                                             intLit(sh))
                          .getResult(),
                      intLit(0x700))
                  .getResult();
          mlir::Value idx =
              rewriter
                  .create<emitc::BitwiseOrOp>(loc, intType, qsIdxByte,
                                              qhHighBits)
                  .getResult();
          // idxoff[l] = (uint16_t)(idx << 3);  (idx*8 byte offset into the u64 grid).
          mlir::Value byteOff =
              rewriter
                  .create<emitc::BitwiseLeftShiftOp>(loc, intType, idx,
                                                     intLit(3))
                  .getResult();
          mlir::Value byteOffU16 =
              rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff)
                  .getResult();
          mlir::Value tmpIdx = rewriter.create<emitc::LiteralOp>(
              loc, rewriter.getIndexType(), std::to_string(l));
          mlir::Value tmpElem =
              rewriter
                  .create<emitc::SubscriptOp>(loc, idxArray,
                                              mlir::ValueRange{tmpIdx})
                  .getResult();
          rewriter.create<emitc::AssignOp>(
              loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(tmpElem),
              byteOffU16);

          // --- sum2/delta term (grid-INDEPENDENT, UNCHANGED) ---------------
          // __riscv_vsetvl_e8m1(8);  (the group is 8 elements; m1 covers it).
          std::string innerSetvlCallee =
              riscvIntrinsicName("vsetvl", 8, "m1", "");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, innerSetvlCallee));
          mlir::Value vl =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{sizeType}, innerSetvlCallee,
                      mlir::ValueRange{sizeLit(groupLanes)})
                  .getResult(0);

          // q8_v = vle8_v_i8m1(q8Group, vl);  (the group's 8 activations).
          std::string i8LoadCallee =
              riscvIntrinsicName("vle", 8, coreLmul, "i8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, i8LoadCallee));
          mlir::Value q8V =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                               i8LoadCallee,
                                               mlir::ValueRange{q8Group, vl})
                  .getResult(0);

          // lsum2 = vwredsum(q8_v, seed=0):  the Σq8 over the 8 activations (the
          // per-group delta term -- iq1_m's four independent group signs make the
          // bsums unusable, so the q8 sum is reduced fresh from the loaded vector).
          // i8 -> i16 widening reduce (|Σ| <= 8*127 = 1016 fits i16).
          std::string seed16Callee = riscvIntrinsicName("vmv_v_x", 16, "m1", "i16");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seed16Callee));
          mlir::Type i16Type = emitc::OpaqueType::get(ctx, "int16_t");
          mlir::Value zero16 =
              rewriter.create<emitc::LiteralOp>(loc, i16Type, "0").getResult();
          mlir::Value seed16 =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m1Type},
                                               seed16Callee,
                                               mlir::ValueRange{zero16,
                                                                sizeLit(1)})
                  .getResult(0);
          std::string q8ReduceCallee = "__riscv_vwredsum_vs_i8m1_i16m1";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, q8ReduceCallee));
          mlir::Value lsum2Acc =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i16m1Type}, q8ReduceCallee,
                      mlir::ValueRange{q8V, seed16, vl})
                  .getResult(0);
          std::string extract16Callee = "__riscv_vmv_x_s_i16m1_i16";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extract16Callee));
          mlir::Value lsum2I16 =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{intType},
                                               extract16Callee,
                                               mlir::ValueRange{lsum2Acc})
                  .getResult(0);

          // sum2[l/2] += lsum2 * delta[l];  (scalar accumulate; order-free).
          int64_t h = l / 2;
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "sum2_delta_accumulate"));
          mlir::Value lsum2Delta =
              rewriter.create<emitc::MulOp>(loc, intType, lsum2I16, delta[l])
                  .getResult();
          mlir::Value lsum2DeltaI32 =
              rewriter.create<emitc::CastOp>(loc, i32Type, lsum2Delta)
                  .getResult();
          mlir::Value sum2Cur =
              rewriter.create<emitc::LoadOp>(loc, i32Type, sum2Var[h])
                  .getResult();
          mlir::Value sum2Next =
              rewriter.create<emitc::AddOp>(loc, i32Type, sum2Cur, lsum2DeltaI32)
                  .getResult();
          rewriter.create<emitc::AssignOp>(loc, sum2Var[h], sum2Next);

          // q8Group += 8 (advance to the next group's 8 activations).
          q8Group =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                        sizeLit(groupLanes))
                  .getResult();
        }

        // --- grid dot via vluxei16, per HALF (h=0,1) -----------------------
        // The half h covers groups {2h, 2h+1} = 16 contiguous q8 activations
        // starting at q8Base + ib*32 + h*16. For each half: gather 2 grid u64
        // entries (idxoff[2h], idxoff[2h+1]) -> i8m1 (16 grid bytes), load the
        // 16 q8, vwmul i16m2, ONE vwredsum -> sum1[h] (assigned ONCE; the 16-lane
        // reduce equals Σ(group 2h) + Σ(group 2h+1), integer-identical to the
        // original two 8-lane reduces summed into sum1[h]).
        mlir::Value grid64 =
            rewriter.create<emitc::CastOp>(loc, i64PtrType, gridArrayName)
                .getResult();
        mlir::Value q8SubBase =
            (ib == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib * subBlock))
                      .getResult();
        for (int64_t h = 0; h < halvesPerSub; ++h) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "ternary_grid_gather_half"));

          // vuint16mf4_t vidx = __riscv_vle16_v_u16mf4(&idxoff[2h], 2);  (index
          // EMUL = (16/64)*1 = 1/4 -> mf4 for the i64m1 gather.)
          mlir::Value idxBaseIndex =
              rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(),
                                                std::to_string(h * groupsPerHalf));
          mlir::Value idxBaseElem =
              rewriter
                  .create<emitc::SubscriptOp>(loc, idxArray,
                                              mlir::ValueRange{idxBaseIndex})
                  .getResult();
          mlir::Value idxBase =
              rewriter
                  .create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", idxBaseElem)
                  .getResult();
          std::string idxLoadCallee =
              riscvIntrinsicName("vle", 16, "mf4", "u16");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, idxLoadCallee));
          mlir::Value vidx =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{u16mf4Type}, idxLoadCallee,
                      mlir::ValueRange{idxBase, sizeLit(groupsPerHalf)})
                  .getResult(0);

          // vint64m1_t g64 = __riscv_vluxei16_v_i64m1(grid64, vidx, 2);  -- the
          // HARDWARE indexed gather of the 2 grid u64 entries (ggml's
          // __riscv_vluxei16), then reinterpret to i8m1 = 16 signed ternary bytes.
          std::string gatherCallee =
              riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", "m1");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, gatherCallee));
          mlir::Value gathered =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i64m1Type}, gatherCallee,
                      mlir::ValueRange{grid64, vidx, sizeLit(groupsPerHalf)})
                  .getResult(0);
          std::string reinterpretCallee = "__riscv_vreinterpret_v_i64m1_i8m1";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reinterpretCallee));
          mlir::Value gridV =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8HalfType},
                                               reinterpretCallee,
                                               mlir::ValueRange{gathered})
                  .getResult(0);

          // q8h = vle8_v_i8m1(q8SubBase + h*16, 16);  (the half's 16 activations).
          mlir::Value q8HalfPtr =
              (h == 0)
                  ? q8SubBase
                  : rewriter
                        .create<emitc::AddOp>(loc, i8PtrType, q8SubBase,
                                              sizeLit(h * halfLanes))
                        .getResult();
          std::string q8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, q8LoadCallee));
          mlir::Value q8H =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i8HalfType}, q8LoadCallee,
                      mlir::ValueRange{q8HalfPtr, sizeLit(halfLanes)})
                  .getResult(0);

          // p = __riscv_vwmul_vv_i16m2(grid, q8h, 16);  (signed widening product;
          // the ternary grid bytes are ALREADY signed {-1,0,+1}, so NO sign apply.)
          std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, wmulCallee));
          mlir::Value product =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i16HalfType}, wmulCallee,
                      mlir::ValueRange{gridV, q8H, sizeLit(halfLanes)})
                  .getResult(0);

          // sum1[h] = __riscv_vmv_x_s(vwredsum_vs_i16m2_i32m1(p, seed=0, 16));
          // -- ONE reduction per half; assigned ONCE (no +=) since the whole
          // half is computed in a single shot.
          std::string seed32Callee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seed32Callee));
          mlir::Value zero32 =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          mlir::Value seed32 =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                               seed32Callee,
                                               mlir::ValueRange{zero32,
                                                                sizeLit(1)})
                  .getResult(0);
          std::string reduceCallee =
              ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reduceCallee));
          mlir::Value sum1Acc =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                      mlir::ValueRange{product, seed32, sizeLit(halfLanes)})
                  .getResult(0);
          std::string extract32Callee = "__riscv_vmv_x_s_i32m1_i32";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extract32Callee));
          mlir::Value sum1H =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                               extract32Callee,
                                               mlir::ValueRange{sum1Acc})
                  .getResult(0);
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "sum1_assign"));
          rewriter.create<emitc::AssignOp>(loc, sum1Var[h], sum1H);
        }

        // ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1;  (groups 0..1)
        // ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1;  (groups 2..3)
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "per_half_scale"));
        mlir::Value scWord = loadU16AsInt(scBase, ib / 2);
        int64_t baseShift = 6 * (ib % 2);
        auto scaleFromBits = [&](int64_t shift) -> mlir::Value {
          mlir::Value field =
              rewriter
                  .create<emitc::BitwiseAndOp>(
                      loc, intType,
                      rewriter
                          .create<emitc::BitwiseRightShiftOp>(
                              loc, intType, scWord, intLit(shift))
                          .getResult(),
                      intLit(7))
                  .getResult();
          return rewriter
              .create<emitc::AddOp>(
                  loc, intType,
                  rewriter.create<emitc::MulOp>(loc, intType, field, intLit(2))
                      .getResult(),
                  intLit(1))
              .getResult();
        };
        mlir::Value ls1 = scaleFromBits(baseShift + 0);
        mlir::Value ls2 = scaleFromBits(baseShift + 3);
        mlir::Value ls1I32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, ls1).getResult();
        mlir::Value ls2I32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, ls2).getResult();

        // sumi1 += sum1[0]*ls1 + sum1[1]*ls2;  sumi2 += sum2[0]*ls1 + sum2[1]*ls2.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sumi_accumulate"));
        auto foldHalves = [&](emitc::VariableOp v0, emitc::VariableOp v1) {
          mlir::Value a =
              rewriter.create<emitc::LoadOp>(loc, i32Type, v0).getResult();
          mlir::Value b =
              rewriter.create<emitc::LoadOp>(loc, i32Type, v1).getResult();
          mlir::Value aScaled =
              rewriter.create<emitc::MulOp>(loc, i32Type, a, ls1I32).getResult();
          mlir::Value bScaled =
              rewriter.create<emitc::MulOp>(loc, i32Type, b, ls2I32).getResult();
          return rewriter
              .create<emitc::AddOp>(loc, i32Type, aScaled, bScaled)
              .getResult();
        };
        mlir::Value sumi1Add = foldHalves(sum1Var[0], sum1Var[1]);
        mlir::Value sumi1Cur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumi1Var).getResult();
        rewriter.create<emitc::AssignOp>(
            loc, sumi1Var,
            rewriter.create<emitc::AddOp>(loc, i32Type, sumi1Cur, sumi1Add)
                .getResult());
        mlir::Value sumi2Add = foldHalves(sum2Var[0], sum2Var[1]);
        mlir::Value sumi2Cur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumi2Var).getResult();
        rewriter.create<emitc::AssignOp>(
            loc, sumi2Var,
            rewriter.create<emitc::AddOp>(loc, i32Type, sumi2Cur, sumi2Add)
                .getResult());
      }

      // sumf = sumf + d * ((float)sumi1 + 0.125f * (float)sumi2);  -- ONE
      // emitc.expression (ggml's single C statement). The 0.125 (IQ1M_DELTA) is
      // applied EXACTLY ONCE here, to (float)sumi2 only; the inner add stays in the
      // expression tree. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate_with_delta"));
      mlir::Value sumi1Final =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumi1Var).getResult();
      mlir::Value sumi2Final =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumi2Var).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      mlir::Value deltaConst =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.125f");
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumi1Float =
            rewriter.create<emitc::CastOp>(loc, floatType, sumi1Final)
                .getResult();
        mlir::Value sumi2Float =
            rewriter.create<emitc::CastOp>(loc, floatType, sumi2Final)
                .getResult();
        mlir::Value deltaScaled =
            rewriter.create<emitc::MulOp>(loc, floatType, deltaConst, sumi2Float)
                .getResult();
        mlir::Value inner =
            rewriter.create<emitc::AddOp>(loc, floatType, sumi1Float,
                                          deltaScaled)
                .getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, inner).getResult();
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm)
                .getResult();
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    }

    // *s = sumf;  (iq1_m applies NO trailing factor -- the store is the bare
    // accumulator; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq1_m block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitTQ2_0Q8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotTQ20Q8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotTQ20Q8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "tq2_0 block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "tq2_0 block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t weightStride = blockDot.getWeightBlockStride();         // 66
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //  0
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        // 64
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//  0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //  4
    int64_t subBlock = 16;                                          // 16x16
    int64_t numSubBlocks = qk / subBlock;                           // 16

    // The vector types for the 2-bit unpack (e8m2 32-lane chunk) and the
    // per-sub-block widening dot (e8m1 16-lane -> i16m2 -> i32m1). The dot uses
    // LMUL=1 (e8m1) so the 16-element sub-block reduce sees all 16 lanes
    // (VLMAX(e8m1) = VLEN/8 = 16 at VLEN >= 128; e8mf2 would cap at 8 and halve
    // the per-sub-block partial sum).
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

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered ternary
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
    // OUTSIDE the loop (mirrors _generic's `float sumf = 0.0f;`).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // Per-super-block base address arithmetic (vx + ib*66, vy + ib*292).
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

      // ---- (A) the 2-bit ternary unpack into aux8[256] (element-ordered) ----
      // For each 32-byte qs chunk (chunk in 0..1) and each 2-bit shift in
      // {0,2,4,6}: aux8[128*chunk + 32*(shift/2) + l] = ((qs[chunk*32+l] >>
      // shift) & 3) - 1 for the 32 lanes l. The `-1` ternary bias is folded PER
      // ELEMENT into the unpack (a vadd.vx of -1 in the i8 domain after the
      // u8->i8 reinterpret), mirroring _generic's per-element subtract. Sub-
      // block s -> aux8[16s:16s+16] pairs with q8[16s:16s+16].
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "unpack_2bit_ternary"));
      std::string unpackSetvl = "__riscv_vsetvl_e8m2";
      for (int64_t chunk = 0; chunk < qk / 128; ++chunk) {
        int64_t qsChunk = chunk * 32; // q2 advances 32 bytes per 128-elem chunk
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, unpackSetvl));
        mlir::Value vlu =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             unpackSetvl,
                                             mlir::ValueRange{sizeLit(32)})
                .getResult(0);
        mlir::Value qsPtr =
            byteOffsetPtr(xb, weightPtrType, qsOffset + qsChunk, u8PtrType);
        std::string loadCallee = "__riscv_vle8_v_u8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value q2 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8m2Type},
                                             loadCallee,
                                             mlir::ValueRange{qsPtr, vlu})
                .getResult(0);
        for (int64_t j = 0; j < 4; ++j) {
          int64_t shift = 2 * j;
          int64_t aChunk = chunk * 128 + j * 32; // aux8 base for this shift
          mlir::Value nib = q2;
          if (shift != 0) {
            std::string srlCallee = "__riscv_vsrl_vx_u8m2";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, srlCallee));
            mlir::Value amt = rewriter.create<emitc::LiteralOp>(
                loc, i32ImmType, std::to_string(shift));
            nib = rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{u8m2Type}, srlCallee,
                          mlir::ValueRange{q2, amt, vlu})
                      .getResult(0);
          }
          std::string andCallee = "__riscv_vand_vx_u8m2";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, andCallee));
          mlir::Value mask3 =
              rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0x03");
          mlir::Value q2bits =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8m2Type},
                                               andCallee,
                                               mlir::ValueRange{nib, mask3, vlu})
                  .getResult(0);
          std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reCallee));
          mlir::Value q2i =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8m2Type},
                                               reCallee,
                                               mlir::ValueRange{q2bits})
                  .getResult(0);
          // aux8 = (q2 & 3) - 1  -- the per-element ternary `-1` bias folded
          // into the unpack (vadd.vx of -1 in the i8 domain; q2 in [0,3] is
          // small so the i8 subtract is exact -> aux8 in [-1,2]).
          std::string biasCallee = "__riscv_vadd_vx_i8m2";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, biasCallee));
          mlir::Value biasImm =
              rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "-1");
          mlir::Value ternary =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i8m2Type}, biasCallee,
                      mlir::ValueRange{q2i, biasImm, vlu})
                  .getResult(0);
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
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, storeCallee));
          rewriter.create<emitc::CallOpaqueOp>(
              loc, mlir::TypeRange{}, storeCallee,
              mlir::ValueRange{dstPtr, ternary, vlu});
        }
      }

      // The q8 quant base.
      mlir::Value q8Base =
          byteOffsetPtr(yb, activationPtrType, q8Offset, i8PtrType);

      // ---- (B) the SINGLE per-super-block integer accumulator ----
      // int sumi = 0;  (the per-super-block ternary*q8 dot; integer add is
      // order-free so the 16 sub-block partial sums fold into one scalar).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar,
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult());

      for (int64_t s = 0; s < numSubBlocks; ++s) {
        // sumi += Σ_{l=0..15} q8[16s+l] * aux8[16s+l]  (the vector widen-reduce;
        // integer / order-free). vle8 i8m1 (16 lanes) x2 -> vwmul_vv i16m2 ->
        // vwredsum_vs into i32m1 lane 0 (seed 0) -> vmv_x_s. LMUL=1 (e8m1) so
        // the 16-element reduce sees all 16 lanes (VLMAX(e8m1)=VLEN/8=16 at
        // VLEN>=128). NO per-sub-block scale multiply -- tq2_0 has no scales.
        mlir::Value sIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(s));
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_dot"));
        std::string dotSetvl = "__riscv_vsetvl_e8m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, dotSetvl));
        mlir::Value vl16 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             dotSetvl,
                                             mlir::ValueRange{sizeLit(subBlock)})
                .getResult(0);
        mlir::Value subOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, sIdx, sizeLit(subBlock));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, i8PtrType, q8Base, subOff)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, i8PtrType, aux8Base, subOff)
                .getResult();
        std::string loadCallee = "__riscv_vle8_v_i8m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value q8v =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8m1Type},
                                             loadCallee,
                                             mlir::ValueRange{q8Ptr, vl16})
                .getResult(0);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value av =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8m1Type},
                                             loadCallee,
                                             mlir::ValueRange{aPtr, vl16})
                .getResult(0);
        std::string mulCallee = "__riscv_vwmul_vv_i16m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, mulCallee));
        mlir::Value p =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m2Type},
                                             mulCallee,
                                             mlir::ValueRange{q8v, av, vl16})
                .getResult(0);
        std::string seedCallee = "__riscv_vmv_v_x_i32m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, seedCallee));
        mlir::Value zeroImm =
            rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0");
        mlir::Value seed =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             seedCallee,
                                             mlir::ValueRange{zeroImm, sizeLit(1)})
                .getResult(0);
        std::string reduceCallee = "__riscv_vwredsum_vs_i16m2_i32m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reduceCallee));
        mlir::Value red =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             reduceCallee,
                                             mlir::ValueRange{p, seed, vl16})
                .getResult(0);
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, extractCallee));
        mlir::Value isuml =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                             extractCallee,
                                             mlir::ValueRange{red})
                .getResult(0);

        // sumi += isuml;  (integer, order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sumi_accumulate"));
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
        mlir::Value sumiNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumiCur, isuml)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
      }

      // ---- (C) the SINGLE-SCALE SCALAR fp32 fold: sumf += (float)sumi * d ----
      // float dy = *(const float *)(yb + 0);  -- the fp32 q8_K activation scale.
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

      // float dx = (float)*(const _Float16 *)(xb + 64);  -- the fp16 tq2_0
      // super-block scale (at the END of block_tq2_0).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dxAddr})
              .getResult(0);
      // float d = dy * dx;  -- the single super-block scale, its OWN product
      // (mirrors _generic's `const float d = y[i].d * fp16(x[i].d);`).
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dy, dx).getResult();

      // sumf = sumf + (float)sumi * d;  -- ONE emitc.expression so the cast +
      // the product + the add render as ggml's single C statement (quants.c:508
      // `sumf += (float) sumi * d`) and track its contraction.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scalar_fold"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto foldExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&foldExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        // (float)sumi  -- the int->float conversion.
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // (float)sumi * d  -- the super-block term.
        mlir::Value term =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, d);
        // sumf + (float)sumi * d  -- the `+=`.
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, term);
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
                                         "tq2_0 block-dot output not a pointer");
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

mlir::LogicalResult VariantToEmitCFunc::emitTQ1_0Q8_KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotTQ10Q8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotTQ10Q8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "tq1_0 block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "tq1_0 block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t weightStride = blockDot.getWeightBlockStride();         // 54
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //  0
    int64_t qhOffset = blockDot.getWeightQhByteOffset();            // 48
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        // 52
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//  0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //  4
    int64_t subBlock = 16;                                          // 16x16
    int64_t numSubBlocks = qk / subBlock;                           // 16

    // ggml's base-3 powers: pow3[l] = 3^l for l in 0..4 (qs) and 0..3 (qh).
    const int64_t pow3[6] = {1, 3, 9, 27, 81, 243};

    // The vector types: the base-3 unpack uses e8m2 (32-lane main) and e8m1
    // (16-lane tail / 4-lane qh), with the widening `*3` doubling to u16m4 /
    // u16m2. The per-sub-block widening dot reuses tq2_0's e8m1 -> i16m2 -> i32m1
    // shape; LMUL=1 (e8m1) so the 16-element reduce sees all 16 lanes
    // (VLMAX(e8m1) = VLEN/8 = 16 at VLEN >= 128).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type u16m4Type = emitc::OpaqueType::get(ctx, "vuint16m4_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type u8m1Type = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type u16m2Type = emitc::OpaqueType::get(ctx, "vuint16m2_t");
    mlir::Type i8m1Type = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16m2Type = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
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

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered ternary
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
    // OUTSIDE the loop (mirrors _generic's `float sumf = 0.0f;`).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // Per-super-block base address arithmetic (vx + ib*54, vy + ib*292).
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

      // ---- (A) the BASE-3 trit unpack into aux8[256] (in q8 index order) ----
      // For each region (main qs, tail qs, qh) and each base-3 digit l, recover
      // the trit: `q = (uint8_t)(byte * pow3[l])` (an 8-bit multiply -- the
      // mod-256 wrap IS the decode), then `xi = ((uint16_t)q * 3) >> 8` (the
      // high base-3 digit, in {0,1,2}), then `xi - 1` (the ternary lane). The
      // aux8 destination index matches _generic's q8 index so aux8[i] pairs
      // contiguously with q8[i]. The decode pipeline at LMUL `lmul`:
      //   vmul.vx u8 (byte * pow3[l])  -- the uint8 wrap, NOT widened
      //   vwmulu.vx u8->u16 (q * 3)    -- widen, the high-digit numerator
      //   vsrl.vx u16 (>> 8)           -- xi in {0,1,2}
      //   vncvt.x.x.w u16->u8          -- narrow (lossless, xi small)
      //   vreinterpret u8->i8 ; vadd.vx i8 (-1)  -- the ternary bias
      //   vse8 i8                      -- store at the q8-ordered aux8 slot
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "unpack_base3_trit"));

      // emitBase3Region: decode `lanes` weight bytes at (weightByteBase) for the
      // base-3 digit `l` into aux8[auxBase .. auxBase+lanes), at LMUL m2 (32-lane
      // main) or m1 (<=16-lane tail / qh). Returns nothing -- it stores aux8.
      auto emitBase3Region = [&](int64_t weightByteBase, int64_t l,
                                 int64_t auxBase, int64_t lanes, bool m2) {
        mlir::Type u8Ty = m2 ? u8m2Type : u8m1Type;
        mlir::Type u16Ty = m2 ? u16m4Type : u16m2Type;
        mlir::Type i8Ty = m2 ? i8m2Type : i8m1Type;
        std::string setvlCallee = m2 ? "__riscv_vsetvl_e8m2"
                                     : "__riscv_vsetvl_e8m1";
        std::string loadCallee = m2 ? "__riscv_vle8_v_u8m2"
                                    : "__riscv_vle8_v_u8m1";
        std::string mulCallee = m2 ? "__riscv_vmul_vx_u8m2"
                                   : "__riscv_vmul_vx_u8m1";
        std::string wmulCallee = m2 ? "__riscv_vwmulu_vx_u16m4"
                                    : "__riscv_vwmulu_vx_u16m2";
        std::string srlCallee = m2 ? "__riscv_vsrl_vx_u16m4"
                                   : "__riscv_vsrl_vx_u16m2";
        std::string ncvtCallee = m2 ? "__riscv_vncvt_x_x_w_u8m2"
                                    : "__riscv_vncvt_x_x_w_u8m1";
        std::string reCallee = m2 ? "__riscv_vreinterpret_v_u8m2_i8m2"
                                  : "__riscv_vreinterpret_v_u8m1_i8m1";
        std::string biasCallee = m2 ? "__riscv_vadd_vx_i8m2"
                                    : "__riscv_vadd_vx_i8m1";
        std::string storeCallee = m2 ? "__riscv_vse8_v_i8m2"
                                     : "__riscv_vse8_v_i8m1";

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, setvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             setvlCallee,
                                             mlir::ValueRange{sizeLit(lanes)})
                .getResult(0);
        mlir::Value srcPtr =
            byteOffsetPtr(xb, weightPtrType, weightByteBase, u8PtrType);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value bytes =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8Ty},
                                             loadCallee,
                                             mlir::ValueRange{srcPtr, vl})
                .getResult(0);
        // q = (uint8_t)(byte * pow3[l])  -- the mandatory mod-256 wrap (8-bit).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, mulCallee));
        mlir::Value pow3Imm = rewriter.create<emitc::LiteralOp>(
            loc, i32ImmType, std::to_string(pow3[l]));
        mlir::Value q =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8Ty},
                                             mulCallee,
                                             mlir::ValueRange{bytes, pow3Imm, vl})
                .getResult(0);
        // (uint16_t)q * 3  -- widen then multiply by 3 (the high base-3 digit
        // numerator). Widening AFTER the wrap, never fused into pow3.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, wmulCallee));
        mlir::Value three =
            rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "3");
        mlir::Value w =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u16Ty},
                                             wmulCallee,
                                             mlir::ValueRange{q, three, vl})
                .getResult(0);
        // xi = (q*3) >> 8  -- the high base-3 digit, in {0,1,2}.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, srlCallee));
        mlir::Value eight =
            rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "8");
        mlir::Value xi16 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u16Ty},
                                             srlCallee,
                                             mlir::ValueRange{w, eight, vl})
                .getResult(0);
        // narrow u16 -> u8 (xi in {0,1,2}, lossless).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, ncvtCallee));
        mlir::Value xi8 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8Ty},
                                             ncvtCallee,
                                             mlir::ValueRange{xi16, vl})
                .getResult(0);
        // reinterpret u8 -> i8.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reCallee));
        mlir::Value xiI =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8Ty},
                                             reCallee,
                                             mlir::ValueRange{xi8})
                .getResult(0);
        // xi - 1  -- the ternary bias (vadd.vx of -1 in the i8 domain; xi in
        // {0,1,2} so the i8 subtract is exact -> aux8 in {-1,0,1}).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, biasCallee));
        mlir::Value biasImm =
            rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "-1");
        mlir::Value ternary =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8Ty},
                                             biasCallee,
                                             mlir::ValueRange{xiI, biasImm, vl})
                .getResult(0);
        mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(auxBase));
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
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, storeCallee));
        rewriter.create<emitc::CallOpaqueOp>(
            loc, mlir::TypeRange{}, storeCallee,
            mlir::ValueRange{dstPtr, ternary, vl});
      };

      // (a) main qs loop: j=0, l 0..4, m 0..31. weight byte qs[m] (32 lanes),
      // aux8[l*32 + m] (q8 index j*5 + l*32 + m = l*32 + m). e8m2 (32 lanes).
      for (int64_t l = 0; l < 5; ++l)
        emitBase3Region(/*weightByteBase=*/qsOffset + 0, l,
                        /*auxBase=*/l * 32, /*lanes=*/32, /*m2=*/true);
      // (b) tail qs loop: j=32, l 0..4, m 0..15. weight byte qs[32+m] (16 lanes),
      // aux8[160 + l*16 + m] (q8 index j*5 + l*16 + m = 160 + l*16 + m). e8m1.
      for (int64_t l = 0; l < 5; ++l)
        emitBase3Region(/*weightByteBase=*/qsOffset + 32, l,
                        /*auxBase=*/160 + l * 16, /*lanes=*/16, /*m2=*/false);
      // (c) qh loop: l 0..3, j 0..3. weight byte qh[j] (4 lanes), aux8[240 +
      // l*4 + j] (q8 index sizeof(qs)*5 + l*sizeof(qh) + j = 240 + l*4 + j).
      // e8m1 with vl=4.
      for (int64_t l = 0; l < 4; ++l)
        emitBase3Region(/*weightByteBase=*/qhOffset + 0, l,
                        /*auxBase=*/240 + l * 4, /*lanes=*/4, /*m2=*/false);

      // The q8 quant base.
      mlir::Value q8Base =
          byteOffsetPtr(yb, activationPtrType, q8Offset, i8PtrType);

      // ---- (B) the SINGLE per-super-block integer accumulator ----
      // int sumi = 0;  (the per-super-block ternary*q8 dot; integer add is
      // order-free so the 16 sub-block partial sums fold into one scalar).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar,
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult());

      for (int64_t s = 0; s < numSubBlocks; ++s) {
        // sumi += Σ_{l=0..15} q8[16s+l] * aux8[16s+l]  (the vector widen-reduce;
        // integer / order-free). vle8 i8m1 (16 lanes) x2 -> vwmul_vv i16m2 ->
        // vwredsum_vs into i32m1 lane 0 (seed 0) -> vmv_x_s. LMUL=1 (e8m1) so
        // the 16-element reduce sees all 16 lanes (VLMAX(e8m1)=VLEN/8=16 at
        // VLEN>=128). NO per-sub-block scale multiply -- tq1_0 has no scales.
        mlir::Value sIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(s));
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_dot"));
        std::string dotSetvl = "__riscv_vsetvl_e8m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, dotSetvl));
        mlir::Value vl16 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             dotSetvl,
                                             mlir::ValueRange{sizeLit(subBlock)})
                .getResult(0);
        mlir::Value subOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, sIdx, sizeLit(subBlock));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, i8PtrType, q8Base, subOff)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, i8PtrType, aux8Base, subOff)
                .getResult();
        std::string loadCallee = "__riscv_vle8_v_i8m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value q8v =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8m1Type},
                                             loadCallee,
                                             mlir::ValueRange{q8Ptr, vl16})
                .getResult(0);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value av =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8m1Type},
                                             loadCallee,
                                             mlir::ValueRange{aPtr, vl16})
                .getResult(0);
        std::string mulCallee = "__riscv_vwmul_vv_i16m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, mulCallee));
        mlir::Value p =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16m2Type},
                                             mulCallee,
                                             mlir::ValueRange{q8v, av, vl16})
                .getResult(0);
        std::string seedCallee = "__riscv_vmv_v_x_i32m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, seedCallee));
        mlir::Value zeroImm =
            rewriter.create<emitc::LiteralOp>(loc, i32ImmType, "0");
        mlir::Value seed =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             seedCallee,
                                             mlir::ValueRange{zeroImm, sizeLit(1)})
                .getResult(0);
        std::string reduceCallee = "__riscv_vwredsum_vs_i16m2_i32m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reduceCallee));
        mlir::Value red =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             reduceCallee,
                                             mlir::ValueRange{p, seed, vl16})
                .getResult(0);
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, extractCallee));
        mlir::Value isuml =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                             extractCallee,
                                             mlir::ValueRange{red})
                .getResult(0);

        // sumi += isuml;  (integer, order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sumi_accumulate"));
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
        mlir::Value sumiNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumiCur, isuml)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
      }

      // ---- (C) the SINGLE-SCALE SCALAR fp32 fold: sumf += (float)sumi * d ----
      // float dx = (float)*(const _Float16 *)(xb + 52);  -- the fp16 tq1_0
      // super-block scale (at the END of block_tq1_0, after qs[48] + qh[4]).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dxAddr})
              .getResult(0);

      // float dy = *(const float *)(yb + 0);  -- the fp32 q8_K activation scale.
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

      // float d = dx * dy;  -- the single super-block scale, its OWN product
      // (mirrors _generic's `(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d)`: the fp16
      // weight scale FIRST, then the fp32 activation scale).
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // sumf = sumf + (float)sumi * d;  -- ONE emitc.expression so the cast +
      // the product + the add render as ggml's single C statement (quants.c:476
      // `sumf += (float) sum * (...)`) and track its contraction.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scalar_fold"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto foldExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&foldExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        // (float)sumi  -- the int->float conversion.
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // (float)sumi * d  -- the super-block term.
        mlir::Value term =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, d);
        // sumf + (float)sumi * d  -- the `+=`.
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, term);
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
                                         "tq1_0 block-dot output not a pointer");
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
