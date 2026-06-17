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

// VariantToEmitCFunc grid-codebook emit methods: iq2_xxs/iq2_xs/iq2_s and
// iq3_xxs/iq3_s (grid + sign-plane decode). Split out of RVVToEmitC.cpp as a
// pure code move; the emitted C is byte-identical.

mlir::LogicalResult VariantToEmitCFunc::emitIQ2XXSQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ2XXSQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ2XXSQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq2_xxs block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq2_xxs block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    // aux1 + the scale `ls = 2*(aux1>>28)+1` + the sign selector `(aux1>>7*l)&127`
    // must be computed in the UNSIGNED domain (ggml reads aux32[1] as a uint32_t)
    // so the >> is a LOGICAL shift -- a signed `int` aux1 with bit 31 set would
    // arithmetic-shift and corrupt the scale/selector.
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t subBlock = blockDot.getSubBlock();                      //  32
    int64_t weightStride = blockDot.getWeightBlockStride();         //  66
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   2
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t numGroups = 4;     // 4 grid groups of 8 elements per sub-block
    int64_t groupLanes = 8;    // 8 grid values per group

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

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
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The GRID codebook is a STRUCTURAL fact off the typed attr (I4 mirror): the
    // 256 uint64 iq2xxs_grid literals, emitted ONCE as a `static const int64_t[256]`
    // decl (rendering ggml's exact hex literals) and read as bytes through a
    // (const int8_t *) cast -- copying ggml's source method (`grid = (const uint8_t
    // *)(iq2xxs_grid + idx)`). Every grid byte is <= 0x2b (43) < 128, so reading it
    // as int8 yields the identical numeric value as ggml's uint8 read.
    llvm::ArrayRef<int64_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const int64_t tcrv_iq2xxs_grid[256] = {";
      for (size_t i = 0; i < grid.size(); ++i) {
        if (i)
          decl += ", ";
        // Render as a hex literal with the unsigned 64-bit suffix so the int64_t
        // initializer carries the exact uint64 bit pattern ggml's table holds.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%016llxULL",
                      static_cast<unsigned long long>(
                          static_cast<uint64_t>(grid[i])));
        decl += buf;
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // The SIGN PLANE is a STRUCTURAL fact off the typed attr (I4 mirror): the 128
    // ksigns_iq2xs entries (values reach 255, beyond int8, carried as i32 in the
    // attr) emitted ONCE as a `static const uint8_t[128]` decl, indexed by the
    // 7-bit sign selector.
    llvm::ArrayRef<int32_t> ksigns = blockDot.getKsigns();
    {
      std::string decl = "static const uint8_t tcrv_iq2xxs_ksigns[128] = {";
      for (size_t i = 0; i < ksigns.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(ksigns[i]) & 0xff);
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // The kmask sign-bit selector {1<<j} is an inline const (NOT a table off the
    // attr -- it is a trivial bit-position vector), emitted as a decl + broadcast
    // load ONCE above the super-block loop.
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_iq2xxs_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

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

    // vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_iq2xxs_kmask, 8);  (ONCE)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "kmask_table_load"));
    mlir::Value kmaskName =
        rewriter.create<emitc::LiteralOp>(loc, u8PtrType, "tcrv_iq2xxs_kmask");
    mlir::Value kmaskVec =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType},
                riscvIntrinsicName("vle", 8, coreLmul, "u8"),
                mlir::ValueRange{kmaskName, sizeLit(groupLanes)})
            .getResult(0);

    // const int8_t *grid_i8 = (const int8_t *)tcrv_iq2xxs_grid;  (byte view of the
    // grid table; grid_i8 + idx*8 addresses the 8 int8 values of entry idx --
    // exactly ggml's `(const uint8_t *)(iq2xxs_grid + idx)` byte address. The cast
    // is explicit: tcrv_iq2xxs_grid is an int64_t[256] array, so its name must be
    // re-cast to a byte pointer before the idx*8 byte arithmetic).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, "tcrv_iq2xxs_grid");
    mlir::Value gridName =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrayName).getResult();

    // Per-super-block base address arithmetic: xb = vx + ibl*66; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The aux1 / scale / sign-selector bitwise ops run in the UNSIGNED domain
    // (uint32_t) so the >> is a LOGICAL shift (ggml's aux32[1] is uint32_t).
    auto uAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseAndOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseOrOp>(loc, uintType, a, b).getResult();
    };
    auto uShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseRightShiftOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uintLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, uintType,
                                               std::to_string(v) + "u");
    };
    // uint32_t x = (uint32_t)a[i];  -- a structured byte load from a `const uint8_t
    // *` then a cast to uint32_t (used to reassemble aux1 from the 2-aligned qs
    // stream alignment-safely; NO `*(uint32_t*)`).
    auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, uintType, u8).getResult();
    };
    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used to read the grid index from the 2-aligned qs stream
    // alignment-safely; NO `*(uint32_t*)`).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
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

      // const uint8_t *qs = xb + 2;  const int8_t *q8 = yb + 4;
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

      // int32_t bsum = 0;  (the integer super-block accumulator, reset per
      // super-block; ggml's per-super-block bsum).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("bsum", opName, role));
      auto bsumVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, bsumVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib32 = 0..7), fully unrolled so each
      // sub-block's scale + grid/sign integer dot accumulate in order.
      for (int64_t ib32 = 0; ib32 < numSubBlocks; ++ib32) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_aux_scale"));
        // const uint8_t *a = qs + ib32*8;  (the 8 aux bytes for this sub-block).
        mlir::Value aBase =
            (ib32 == 0)
                ? qsBase
                : rewriter
                      .create<emitc::AddOp>(loc, u8PtrType, qsBase,
                                            sizeLit(ib32 * 8))
                      .getResult();

        // uint32_t aux1 = a[4] | a[5]<<8 | a[6]<<16 | a[7]<<24;  (reassembled
        // little-endian from 4 byte loads in the UNSIGNED domain -- alignment-safe;
        // the qs stream is 2-aligned, and the unsigned type makes the >> logical).
        mlir::Value aux1 = loadByteAsUint(aBase, 4);
        aux1 = uOr(aux1, uShl(loadByteAsUint(aBase, 5), uintLit(8)));
        aux1 = uOr(aux1, uShl(loadByteAsUint(aBase, 6), uintLit(16)));
        aux1 = uOr(aux1, uShl(loadByteAsUint(aBase, 7), uintLit(24)));

        // int ls = 2*(aux1 >> 28) + 1;  (the per-sub-block 4-bit scale, [1,31]); the
        // (aux1 >> 28) is computed in uint32_t (logical shift) then cast to int for
        // the integer scale arithmetic and the later sumi*ls fold.
        mlir::Value lsTop =
            rewriter.create<emitc::CastOp>(loc, intType, uShr(aux1, uintLit(28)))
                .getResult();
        mlir::Value ls =
            rewriter
                .create<emitc::AddOp>(
                    loc, intType,
                    rewriter
                        .create<emitc::MulOp>(loc, intType, lsTop, intLit(2))
                        .getResult(),
                    intLit(1))
                .getResult();

        // int32_t sumi = 0;  (the chained i32m1 reduction seed for the 4 groups;
        // seeded into the first vwredsum, then carried as the scalar seed of the
        // next -- integer add is order-free).
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, seedCallee));
        mlir::Value zeroSeed =
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
        mlir::Value sumiAcc =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             seedCallee,
                                             mlir::ValueRange{zeroSeed,
                                                              sizeLit(1)})
                .getResult(0);

        mlir::Value q8Group =
            (ib32 == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib32 * subBlock))
                      .getResult();

        for (int64_t l = 0; l < numGroups; ++l) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "grid_sign_group"));
          // int idx = a[l];  (the grid index byte for this group).
          mlir::Value idx = loadByteAsInt(aBase, l);
          // int signs = tcrv_iq2xxs_ksigns[(aux1 >> 7*l) & 127];  (the shift is
          // logical in the uint32_t domain; cast the [0,127] selector to int for the
          // table subscript).
          mlir::Value signSel = rewriter
                                    .create<emitc::CastOp>(
                                        loc, intType,
                                        uAnd(uShr(aux1, uintLit(7 * l)),
                                             uintLit(127)))
                                    .getResult();
          mlir::Value ksignsName = rewriter.create<emitc::LiteralOp>(
              loc, u8PtrType, "tcrv_iq2xxs_ksigns");
          mlir::Value signsElem =
              rewriter
                  .create<emitc::SubscriptOp>(
                      loc,
                      llvm::cast<mlir::TypedValue<emitc::PointerType>>(
                          ksignsName),
                      signSel)
                  .getResult();
          mlir::Value signsU8 =
              rewriter.create<emitc::LoadOp>(loc, constU8Type, signsElem)
                  .getResult();
          mlir::Value signs =
              rewriter.create<emitc::CastOp>(loc, intType, signsU8).getResult();

          // __riscv_vsetvl_e8m1(8);  (the group is 8 elements; m1 covers it).
          std::string innerSetvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, innerSetvlCallee));
          mlir::Value vl =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                               innerSetvlCallee,
                                               mlir::ValueRange{sizeLit(groupLanes)})
                  .getResult(0);

          // const int8_t *grid = grid_i8 + idx*8;  (indexed pointer arith).
          mlir::Value idxLane =
              rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(8))
                  .getResult();
          mlir::Value gridPtr =
              rewriter.create<emitc::AddOp>(loc, i8PtrType, gridName, idxLane)
                  .getResult();
          // grid_v = vle8_v_i8m1(grid, vl);  q8_v = vle8_v_i8m1(q8Group, vl);
          std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, i8LoadCallee));
          mlir::Value gridV =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                               i8LoadCallee,
                                               mlir::ValueRange{gridPtr, vl})
                  .getResult(0);
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, i8LoadCallee));
          mlir::Value q8V =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                               i8LoadCallee,
                                               mlir::ValueRange{q8Group, vl})
                  .getResult(0);

          // sign-bit mask: m = vmsne(vand(vmv(signs), kmask), 0).
          std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, bcastCallee));
          mlir::Value signsBcast =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                               bcastCallee,
                                               mlir::ValueRange{signs, vl})
                  .getResult(0);
          std::string andCallee = "__riscv_vand_vv_u8m1";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, andCallee));
          mlir::Value signBits =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{u8CoreType}, andCallee,
                      mlir::ValueRange{signsBcast, kmaskVec, vl})
                  .getResult(0);
          std::string msneCallee =
              riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, msneCallee));
          mlir::Value signMask =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{maskType}, msneCallee,
                      mlir::ValueRange{signBits, intLit(0), vl})
                  .getResult(0);

          // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign).
          std::string negCallee = "__riscv_vneg_v_i8m1";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, negCallee));
          mlir::Value gridNeg =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                               negCallee,
                                               mlir::ValueRange{gridV, vl})
                  .getResult(0);
          std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, mergeCallee));
          mlir::Value gridSigned =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i8CoreType}, mergeCallee,
                      mlir::ValueRange{gridV, gridNeg, signMask, vl})
                  .getResult(0);

          // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
          // <= 43*127 = 5461 < 32767, fits i16).
          std::string wmulCallee =
              ("__riscv_vwmul_vv_i16" + wideLmul).str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, wmulCallee));
          mlir::Value product =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i16WideType}, wmulCallee,
                      mlir::ValueRange{gridSigned, q8V, vl})
                  .getResult(0);

          // sumiAcc = vwredsum(product, sumiAcc, vl);  (chain the i32 reduction;
          // integer add is order-free, so seeding from the running accumulator is
          // byte-exact across the 4 groups).
          std::string reduceCallee =
              ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reduceCallee));
          sumiAcc =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                      mlir::ValueRange{product, sumiAcc, vl})
                  .getResult(0);

          // q8Group += 8 (advance to the next group's 8 activations).
          q8Group =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                        sizeLit(groupLanes))
                  .getResult();
        }

        // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (extract after the 4
        // groups -- the sub-block integer dot).
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, extractCallee));
        mlir::Value sumi =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                             extractCallee,
                                             mlir::ValueRange{sumiAcc})
                .getResult(0);

        // bsum = bsum + sumi * ls;  (integer accumulation; order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, ls).getResult();
        mlir::Value sumiLs =
            rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32).getResult();
        mlir::Value bsumNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, bsumCur, sumiLs)
                .getResult();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("bsum", opName, role));
        rewriter.create<emitc::AssignOp>(loc, bsumVar, bsumNext);
      }

      // sumf = sumf + d * (float)bsum;  -- ONE emitc.expression so it renders as
      // ggml's single C statement and the compiler fuses the SAME FMA under
      // -ffp-contract=on/default. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value bsumFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value bsumFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, bsumFinal).getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, bsumFloat)
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

    // *s = 0.125f * sumf;  (the iq2_xxs trailing 1/8 factor, a SEPARATE statement
    // OUTSIDE the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq2_xxs block-dot output not a pointer");
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

    valueMap[blockDot.getResult()] = scaled;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitIQ3XXSQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ3XXSQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ3XXSQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq3_xxs block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq3_xxs block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    // aux32 + the scale `ls = 2*(aux32>>28)+1` + the sign selector `(aux32>>7*l)&127`
    // must be computed in the UNSIGNED domain (ggml reads aux32 as a uint32_t) so the
    // >> is a LOGICAL shift -- a signed `int` aux32 with bit 31 set would arithmetic-
    // shift and corrupt the scale/selector (the iq2_xxs hardware-bisected bug).
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t subBlock = blockDot.getSubBlock();                      //  32
    int64_t weightStride = blockDot.getWeightBlockStride();         //  98
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   2
    int64_t gasOffset = blockDot.getWeightGasByteOffset();          //  66
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t numGroups = 4;     // 4 sign groups of 8 elements per sub-block
    int64_t entryLanes = 4;    // 4 grid values per grid-of-4 entry (the iq3 delta)
    int64_t indicesPerSubBlock = 8; // 8 grid index bytes per sub-block (2 per group)

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

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
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The GRID-of-4 codebook is a STRUCTURAL fact off the typed attr (I4 mirror): the
    // 256 uint32 iq3xxs_grid literals, emitted ONCE as a `static const uint32_t[256]`
    // decl (rendering ggml's exact hex literals) and read as bytes through a
    // (const int8_t *) cast -- copying ggml's source method (`grid = (const uint8_t
    // *)(iq3xxs_grid + idx)`). Every grid byte is <= 0x3e (62) < 128, so reading it
    // as int8 yields the identical numeric value as ggml's uint8 read. Each entry is
    // 4 bytes, so grid_i8 + idx*4 addresses the 4 int8 values of entry idx.
    llvm::ArrayRef<int32_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const uint32_t tcrv_iq3xxs_grid[256] = {";
      for (size_t i = 0; i < grid.size(); ++i) {
        if (i)
          decl += ", ";
        char buf[24];
        std::snprintf(buf, sizeof(buf), "0x%08xU",
                      static_cast<unsigned>(
                          static_cast<uint32_t>(grid[i])));
        decl += buf;
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // The SIGN PLANE is a STRUCTURAL fact off the typed attr (I4 mirror): the 128
    // ksigns_iq2xs entries (values reach 255, beyond int8, carried as i32 in the
    // attr) emitted ONCE as a `static const uint8_t[128]` decl, indexed by the
    // 7-bit sign selector. REUSED verbatim from iq2_xxs.
    llvm::ArrayRef<int32_t> ksigns = blockDot.getKsigns();
    {
      std::string decl = "static const uint8_t tcrv_iq3xxs_ksigns[128] = {";
      for (size_t i = 0; i < ksigns.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(ksigns[i]) & 0xff);
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // The kmask sign-bit selector {1<<j} is an inline const (NOT a table off the
    // attr -- it is a trivial bit-position vector), emitted as a decl + broadcast
    // load ONCE above the super-block loop. kmaskLo = vle8(kmask, 4) carries lanes
    // 0..3 ({1,2,4,8}) for grid1; kmaskHi = vle8(kmask+4, 4) carries lanes 4..7
    // ({16,32,64,128}) for grid2 (the iq3 grid-of-4 splits one sign group across two
    // 4-lane passes, so the SAME signs byte is masked with the lo-4 / hi-4 bits).
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_iq3xxs_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

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

    // vuint8m1_t kmaskLo = vle8(tcrv_iq3xxs_kmask,     4);  (ONCE)
    // vuint8m1_t kmaskHi = vle8(tcrv_iq3xxs_kmask + 4, 4);  (ONCE)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "kmask_table_load"));
    mlir::Value kmaskName =
        rewriter.create<emitc::LiteralOp>(loc, u8PtrType, "tcrv_iq3xxs_kmask");
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
    mlir::Value kmaskLo =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType}, u8LoadCallee,
                mlir::ValueRange{kmaskName, sizeLit(entryLanes)})
            .getResult(0);
    mlir::Value kmaskHiName =
        rewriter.create<emitc::AddOp>(loc, u8PtrType, kmaskName, sizeLit(4))
            .getResult();
    mlir::Value kmaskHi =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType}, u8LoadCallee,
                mlir::ValueRange{kmaskHiName, sizeLit(entryLanes)})
            .getResult(0);

    // const int8_t *grid_i8 = (const int8_t *)tcrv_iq3xxs_grid;  (byte view of the
    // grid table; grid_i8 + idx*4 addresses the 4 int8 values of entry idx --
    // exactly ggml's `(const uint8_t *)(iq3xxs_grid + idx)` byte address. The cast
    // is explicit: tcrv_iq3xxs_grid is a uint32_t[256] array, so its name must be
    // re-cast to a byte pointer before the idx*4 byte arithmetic).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u32PtrType, "tcrv_iq3xxs_grid");
    mlir::Value gridName =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrayName).getResult();

    // Per-super-block base address arithmetic: xb = vx + ibl*98; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The aux32 / scale / sign-selector bitwise ops run in the UNSIGNED domain
    // (uint32_t) so the >> is a LOGICAL shift (ggml's aux32 is uint32_t).
    auto uAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseAndOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseOrOp>(loc, uintType, a, b).getResult();
    };
    auto uShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseRightShiftOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uintLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, uintType,
                                               std::to_string(v) + "u");
    };
    // uint32_t x = (uint32_t)a[i];  -- a structured byte load from a `const uint8_t
    // *` then a cast to uint32_t (used to reassemble aux32 from the 2-aligned gas
    // stream alignment-safely; NO `*(uint32_t*)`).
    auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, uintType, u8).getResult();
    };
    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used to read each grid index from the q3 stream alignment-safely).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
    };

    // ONE grid-of-4 pass: load 4 grid bytes of `idx`, apply the per-lane sign mask
    // (the same `signs` byte vand'd with `kmaskVec` -- {1,2,4,8} for lanes 0..3 or
    // {16,32,64,128} for lanes 4..7), widening-multiply against the 4 q8 lanes at
    // q8Ptr, and chain the i32 vwredsum into `sumiAcc`. This is iq2_xxs's group body
    // at width 4 -- the GRID + SIGN mechanism reused, the grid-of-4 the only delta.
    auto gridOf4Pass = [&](int64_t idxConstant, mlir::Value qgBase,
                           mlir::Value signs, mlir::Value kmaskVec,
                           mlir::Value q8Ptr,
                           mlir::Value sumiAcc) -> mlir::Value {
      // int idx = qg[idxConstant];  (the grid index byte for this 4-lane pass).
      mlir::Value idx = loadByteAsInt(qgBase, idxConstant);

      // __riscv_vsetvl_e8m1(4);  (the pass is 4 elements; m1 covers it).
      std::string innerSetvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value vl =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(entryLanes)})
              .getResult(0);

      // const int8_t *grid = grid_i8 + idx*4;  (indexed pointer arith, grid-of-4).
      mlir::Value idxLane =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(entryLanes))
              .getResult();
      mlir::Value gridPtr =
          rewriter.create<emitc::AddOp>(loc, i8PtrType, gridName, idxLane)
              .getResult();
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, i8LoadCallee));
      mlir::Value gridV =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                           i8LoadCallee,
                                           mlir::ValueRange{gridPtr, vl})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, i8LoadCallee));
      mlir::Value q8V =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                           i8LoadCallee,
                                           mlir::ValueRange{q8Ptr, vl})
              .getResult(0);

      // sign-bit mask: m = vmsne(vand(vmv(signs), kmaskVec), 0).
      std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, bcastCallee));
      mlir::Value signsBcast =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                           bcastCallee,
                                           mlir::ValueRange{signs, vl})
              .getResult(0);
      std::string andCallee = "__riscv_vand_vv_u8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, andCallee));
      mlir::Value signBits =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{u8CoreType}, andCallee,
                  mlir::ValueRange{signsBcast, kmaskVec, vl})
              .getResult(0);
      std::string msneCallee =
          riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, msneCallee));
      mlir::Value signMask =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{maskType}, msneCallee,
                  mlir::ValueRange{signBits, intLit(0), vl})
              .getResult(0);

      // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign).
      std::string negCallee = "__riscv_vneg_v_i8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, negCallee));
      mlir::Value gridNeg =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                           negCallee,
                                           mlir::ValueRange{gridV, vl})
              .getResult(0);
      std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mergeCallee));
      mlir::Value gridSigned =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{i8CoreType}, mergeCallee,
                  mlir::ValueRange{gridV, gridNeg, signMask, vl})
              .getResult(0);

      // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
      // <= 62*127 = 7874 < 32767, fits i16).
      std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, wmulCallee));
      mlir::Value product =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{i16WideType}, wmulCallee,
                  mlir::ValueRange{gridSigned, q8V, vl})
              .getResult(0);

      // sumiAcc = vwredsum(product, sumiAcc, vl);  (chain the i32 reduction;
      // integer add is order-free).
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                       reduceCallee,
                                       mlir::ValueRange{product, sumiAcc, vl})
          .getResult(0);
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

      // const uint8_t *q3 = xb + 2;  const uint8_t *gas = xb + 66;  const int8_t
      // *q8 = yb + 4;  (q3 = the 64 grid index bytes, gas = the 32 aux bytes -- the
      // iq3_xxs SEPARATE qs[96] regions, unlike iq2_xxs's interleaved aux pair).
      mlir::Value q3Base0 = xb;
      if (qsOffset != 0)
        q3Base0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qsOffset));
      mlir::Value q3Base =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, q3Base0).getResult();
      mlir::Value gasBase0 = xb;
      if (gasOffset != 0)
        gasBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                 sizeLit(gasOffset));
      mlir::Value gasBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, gasBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();

      // int32_t bsum = 0;  (the integer super-block accumulator, reset per
      // super-block; ggml's per-super-block bsum).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("bsum", opName, role));
      auto bsumVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, bsumVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib32 = 0..7), fully unrolled so each
      // sub-block's scale + grid/sign integer dot accumulate in order.
      for (int64_t ib32 = 0; ib32 < numSubBlocks; ++ib32) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_aux_scale"));
        // const uint8_t *a = gas + ib32*4;  (the 4 aux bytes for this sub-block).
        mlir::Value aBase =
            (ib32 == 0)
                ? gasBase
                : rewriter
                      .create<emitc::AddOp>(loc, u8PtrType, gasBase,
                                            sizeLit(ib32 * 4))
                      .getResult();

        // uint32_t aux32 = a[0] | a[1]<<8 | a[2]<<16 | a[3]<<24;  (reassembled
        // little-endian from 4 byte loads in the UNSIGNED domain -- alignment-safe;
        // the gas stream is 2-aligned, and the unsigned type makes the >> logical).
        mlir::Value aux32 = loadByteAsUint(aBase, 0);
        aux32 = uOr(aux32, uShl(loadByteAsUint(aBase, 1), uintLit(8)));
        aux32 = uOr(aux32, uShl(loadByteAsUint(aBase, 2), uintLit(16)));
        aux32 = uOr(aux32, uShl(loadByteAsUint(aBase, 3), uintLit(24)));

        // int ls = 2*(aux32 >> 28) + 1;  (the per-sub-block 4-bit scale, [1,31]); the
        // (aux32 >> 28) is computed in uint32_t (logical shift) then cast to int for
        // the integer scale arithmetic and the later sumi*ls fold.
        mlir::Value lsTop =
            rewriter.create<emitc::CastOp>(loc, intType, uShr(aux32, uintLit(28)))
                .getResult();
        mlir::Value ls =
            rewriter
                .create<emitc::AddOp>(
                    loc, intType,
                    rewriter
                        .create<emitc::MulOp>(loc, intType, lsTop, intLit(2))
                        .getResult(),
                    intLit(1))
                .getResult();

        // int32_t sumi = 0;  (the chained i32m1 reduction seed for the 4 groups;
        // seeded into the first vwredsum, then carried as the scalar seed of the
        // next -- integer add is order-free).
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, seedCallee));
        mlir::Value zeroSeed =
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
        mlir::Value sumiAcc =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             seedCallee,
                                             mlir::ValueRange{zeroSeed,
                                                              sizeLit(1)})
                .getResult(0);

        // const uint8_t *qg = q3 + ib32*8;  (the 8 grid index bytes for this
        // sub-block; 2 indices per sign group).
        mlir::Value qgBase =
            (ib32 == 0)
                ? q3Base
                : rewriter
                      .create<emitc::AddOp>(loc, u8PtrType, q3Base,
                                            sizeLit(ib32 * indicesPerSubBlock))
                      .getResult();

        mlir::Value q8Group =
            (ib32 == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib32 * subBlock))
                      .getResult();

        for (int64_t l = 0; l < numGroups; ++l) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "grid_sign_group"));
          // int signs = tcrv_iq3xxs_ksigns[(aux32 >> 7*l) & 127];  (the shift is
          // logical in the uint32_t domain; cast the [0,127] selector to int for the
          // table subscript). REUSES the ksigns sign plane.
          mlir::Value signSel = rewriter
                                    .create<emitc::CastOp>(
                                        loc, intType,
                                        uAnd(uShr(aux32, uintLit(7 * l)),
                                             uintLit(127)))
                                    .getResult();
          mlir::Value ksignsName = rewriter.create<emitc::LiteralOp>(
              loc, u8PtrType, "tcrv_iq3xxs_ksigns");
          mlir::Value signsElem =
              rewriter
                  .create<emitc::SubscriptOp>(
                      loc,
                      llvm::cast<mlir::TypedValue<emitc::PointerType>>(
                          ksignsName),
                      signSel)
                  .getResult();
          mlir::Value signsU8 =
              rewriter.create<emitc::LoadOp>(loc, constU8Type, signsElem)
                  .getResult();
          mlir::Value signs =
              rewriter.create<emitc::CastOp>(loc, intType, signsU8).getResult();

          // q8 pointers for the two 4-lane passes: pass A reads q8Group[0..3]
          // (kmask lanes 0..3), pass B reads q8Group[4..7] (kmask lanes 4..7).
          mlir::Value q8PassB =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group, sizeLit(4))
                  .getResult();

          // pass A: grid1 = grid_i8 + qg[2l+0]*4 vs q8[0..3], kmaskLo {1,2,4,8}.
          sumiAcc = gridOf4Pass(2 * l + 0, qgBase, signs, kmaskLo, q8Group,
                                sumiAcc);
          // pass B: grid2 = grid_i8 + qg[2l+1]*4 vs q8[4..7], kmaskHi {16,32,64,128}.
          sumiAcc = gridOf4Pass(2 * l + 1, qgBase, signs, kmaskHi, q8PassB,
                                sumiAcc);

          // q8Group += 8 (advance to the next group's 8 activations; each sign
          // group covers 8 elements = two grid-of-4 passes).
          int64_t groupElems = subBlock / numGroups; // 32/4 = 8
          q8Group =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                        sizeLit(groupElems))
                  .getResult();
        }

        // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (extract after the 4
        // groups / 8 passes -- the sub-block integer dot).
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, extractCallee));
        mlir::Value sumi =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                             extractCallee,
                                             mlir::ValueRange{sumiAcc})
                .getResult(0);

        // bsum = bsum + sumi * ls;  (integer accumulation; order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, ls).getResult();
        mlir::Value sumiLs =
            rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32).getResult();
        mlir::Value bsumNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, bsumCur, sumiLs)
                .getResult();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("bsum", opName, role));
        rewriter.create<emitc::AssignOp>(loc, bsumVar, bsumNext);
      }

      // sumf = sumf + d * (float)bsum;  -- ONE emitc.expression so it renders as
      // ggml's single C statement and the compiler fuses the SAME FMA under
      // -ffp-contract=on/default. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value bsumFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value bsumFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, bsumFinal).getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, bsumFloat)
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

    // *s = 0.25f * sumf;  (the iq3_xxs trailing 1/4 factor -- DIFFERENT from
    // iq2_xxs's 0.125f -- a SEPARATE statement OUTSIDE the accumulate expression;
    // structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq3_xxs block-dot output not a pointer");
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

    valueMap[blockDot.getResult()] = scaled;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitIQ3SQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ3SQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ3SQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq3_s block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq3_s block-dot ABI operand unmapped");

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
    int64_t weightStride = blockDot.getWeightBlockStride();         // 110
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   2
    int64_t qhOffset = blockDot.getWeightQhByteOffset();            //  66
    int64_t signsOffset = blockDot.getWeightSignsByteOffset();      //  74
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();    // 106
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t numGroups = 4;     // 4 sign groups of 8 elements per sub-block
    int64_t entryLanes = 4;    // 4 grid values per grid-of-4 entry (the iq3 grid)
    int64_t indicesPerSubBlock = 8; // 8 grid index bytes per sub-block (2 per group)
    int64_t signsPerSubBlock = 4;   // 4 explicit sign bytes per sub-block

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

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
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The GRID-of-4 codebook is a STRUCTURAL fact off the typed attr (I4 mirror): the
    // 512 uint32 iq3s_grid literals, emitted ONCE as a `static const uint32_t[512]`
    // decl (rendering ggml's exact hex literals) and read as bytes through a
    // (const int8_t *) cast -- copying ggml's source method (`grid = (const uint8_t
    // *)(iq3s_grid + idx)`). Every grid byte is <= 15 < 128, so reading it as int8
    // yields the identical numeric value as ggml's uint8 read. Each entry is 4 bytes,
    // so grid_i8 + idx*4 addresses the 4 int8 values of entry idx.
    llvm::ArrayRef<int32_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const uint32_t tcrv_iq3s_grid[512] = {";
      for (size_t i = 0; i < grid.size(); ++i) {
        if (i)
          decl += ", ";
        char buf[24];
        std::snprintf(buf, sizeof(buf), "0x%08xU",
                      static_cast<unsigned>(static_cast<uint32_t>(grid[i])));
        decl += buf;
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // The kmask sign-bit selector {1<<j} is an inline const (NOT a table off the
    // attr -- iq3_s carries no ksigns plane; the signs are an explicit memory region).
    // kmaskLo = vle8(kmask, 4) carries lanes 0..3 ({1,2,4,8}) for the grid1 pass;
    // kmaskHi = vle8(kmask+4, 4) carries lanes 4..7 ({16,32,64,128}) for the grid2
    // pass (the iq3 grid-of-4 splits one sign group across two 4-lane passes, so the
    // SAME signs byte is masked with the lo-4 / hi-4 bits -- exactly iq3_xxs).
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_iq3s_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

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

    // vuint8m1_t kmaskLo = vle8(tcrv_iq3s_kmask,     4);  (ONCE)
    // vuint8m1_t kmaskHi = vle8(tcrv_iq3s_kmask + 4, 4);  (ONCE)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "kmask_table_load"));
    mlir::Value kmaskName =
        rewriter.create<emitc::LiteralOp>(loc, u8PtrType, "tcrv_iq3s_kmask");
    std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
    mlir::Value kmaskLo =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType}, u8LoadCallee,
                mlir::ValueRange{kmaskName, sizeLit(entryLanes)})
            .getResult(0);
    mlir::Value kmaskHiName =
        rewriter.create<emitc::AddOp>(loc, u8PtrType, kmaskName, sizeLit(4))
            .getResult();
    mlir::Value kmaskHi =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType}, u8LoadCallee,
                mlir::ValueRange{kmaskHiName, sizeLit(entryLanes)})
            .getResult(0);

    // const int8_t *grid_i8 = (const int8_t *)tcrv_iq3s_grid;  (byte view of the grid
    // table; grid_i8 + idx*4 addresses the 4 int8 values of entry idx -- exactly
    // ggml's `(const uint8_t *)(iq3s_grid + idx)` byte address).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type u32PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, u32PtrType, "tcrv_iq3s_grid");
    mlir::Value gridName =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrayName).getResult();

    // Per-super-block base address arithmetic: xb = vx + ibl*110; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used for each grid index byte, qh byte, sign byte, and scale byte;
    // alignment-safe and the values are small/positive so no sign-extension hazard).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
    };

    // ONE grid-of-4 pass: with the grid index ALREADY assembled (qh injected) as an
    // int Value `idx`, load the 4 grid bytes of `idx`, apply the per-lane sign mask
    // (the explicit `signs` byte vand'd with `kmaskVec` -- {1,2,4,8} for lanes 0..3 or
    // {16,32,64,128} for lanes 4..7), widening-multiply against the 4 q8 lanes at
    // q8Ptr, and chain the i32 vwredsum into `sumiAcc`. This is iq3_xxs's grid-of-4
    // pass; the only iq3_s delta is that the index is passed in pre-assembled (the qh
    // injection differs between the two passes, so it is computed outside the helper).
    auto gridOf4Pass = [&](mlir::Value idx, mlir::Value signs,
                           mlir::Value kmaskVec, mlir::Value q8Ptr,
                           mlir::Value sumiAcc) -> mlir::Value {
      // __riscv_vsetvl_e8m1(4);  (the pass is 4 elements; m1 covers it).
      std::string innerSetvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value vl =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(entryLanes)})
              .getResult(0);

      // const int8_t *grid = grid_i8 + idx*4;  (indexed pointer arith, grid-of-4).
      mlir::Value idxLane =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(entryLanes))
              .getResult();
      mlir::Value gridPtr =
          rewriter.create<emitc::AddOp>(loc, i8PtrType, gridName, idxLane)
              .getResult();
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, i8LoadCallee));
      mlir::Value gridV =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                           i8LoadCallee,
                                           mlir::ValueRange{gridPtr, vl})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, i8LoadCallee));
      mlir::Value q8V =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                           i8LoadCallee,
                                           mlir::ValueRange{q8Ptr, vl})
              .getResult(0);

      // sign-bit mask: m = vmsne(vand(vmv(signs), kmaskVec), 0).
      std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, bcastCallee));
      mlir::Value signsBcast =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                           bcastCallee,
                                           mlir::ValueRange{signs, vl})
              .getResult(0);
      std::string andCallee = "__riscv_vand_vv_u8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, andCallee));
      mlir::Value signBits =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{u8CoreType}, andCallee,
                  mlir::ValueRange{signsBcast, kmaskVec, vl})
              .getResult(0);
      std::string msneCallee =
          riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, msneCallee));
      mlir::Value signMask =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{maskType}, msneCallee,
                  mlir::ValueRange{signBits, intLit(0), vl})
              .getResult(0);

      // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign).
      std::string negCallee = "__riscv_vneg_v_i8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, negCallee));
      mlir::Value gridNeg =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                           negCallee,
                                           mlir::ValueRange{gridV, vl})
              .getResult(0);
      std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mergeCallee));
      mlir::Value gridSigned =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{i8CoreType}, mergeCallee,
                  mlir::ValueRange{gridV, gridNeg, signMask, vl})
              .getResult(0);

      // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
      // <= 15*127 = 1905 < 32767, fits i16).
      std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, wmulCallee));
      mlir::Value product =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{i16WideType}, wmulCallee,
                  mlir::ValueRange{gridSigned, q8V, vl})
              .getResult(0);

      // sumiAcc = vwredsum(product, sumiAcc, vl);  (chain the i32 reduction;
      // integer add is order-free).
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                       reduceCallee,
                                       mlir::ValueRange{product, sumiAcc, vl})
          .getResult(0);
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

      // const uint8_t *qs = xb + 2;  (64 grid index bytes)  const uint8_t *qh =
      // xb + 66;  const uint8_t *sgn = xb + 74;  const uint8_t *sc = xb + 106;
      // const int8_t *q8 = yb + 4;  -- the iq3_s SEPARATE regions (signs is a
      // dedicated 32-byte array, NOT inside qs, unlike iq2_s).
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
      mlir::Value sgnBase0 = xb;
      if (signsOffset != 0)
        sgnBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                 sizeLit(signsOffset));
      mlir::Value sgnBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, sgnBase0).getResult();
      mlir::Value scBase0 = xb;
      if (scalesOffset != 0)
        scBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(scalesOffset));
      mlir::Value scBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, scBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();

      // int32_t bsum = 0;  (the integer super-block accumulator, reset per
      // super-block; ggml's per-super-block bsum).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("bsum", opName, role));
      auto bsumVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, bsumVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib32 = 0..7), fully unrolled. Each sub-block
      // reads scales[ib32/2] (low nibble if ib32 even, high nibble if odd), the qh
      // byte qh[ib32], 8 grid index bytes qs[ib32*8 + 2l + {0,1}], 4 explicit sign
      // bytes sgn[ib32*4 + l], and 32 q8 activations.
      for (int64_t ib32 = 0; ib32 < numSubBlocks; ++ib32) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_explicit_scale"));

        // int sc = sc_base[ib32/2];  ls = ib32 even ? 2*(sc & 0xf)+1 : 2*(sc>>4)+1.
        // ggml packs two 4-bit scales per byte of scales[QK_K/64]: sub-blocks
        // (2k, 2k+1) share scales[k] (low/high nibble).
        mlir::Value scByte = loadByteAsInt(scBase, ib32 / 2);
        mlir::Value nibble =
            (ib32 % 2 == 0)
                ? rewriter
                      .create<emitc::BitwiseAndOp>(loc, intType, scByte,
                                                   intLit(15))
                      .getResult()
                : rewriter
                      .create<emitc::BitwiseRightShiftOp>(loc, intType, scByte,
                                                          intLit(4))
                      .getResult();
        mlir::Value ls =
            rewriter
                .create<emitc::AddOp>(
                    loc, intType,
                    rewriter
                        .create<emitc::MulOp>(loc, intType, nibble, intLit(2))
                        .getResult(),
                    intLit(1))
                .getResult();

        // int qhb = qh[ib32];  (the per-sub-block qh-bit plane byte; its bits inject
        // bit 8 of each of the 8 grid indices in this sub-block).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_plane_byte"));
        mlir::Value qhByte = loadByteAsInt(qhBase, ib32);

        // int32_t sumi = 0;  (the chained i32m1 reduction seed for the 4 groups;
        // integer add is order-free).
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, seedCallee));
        mlir::Value zeroSeed =
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
        mlir::Value sumiAcc =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                             seedCallee,
                                             mlir::ValueRange{zeroSeed,
                                                              sizeLit(1)})
                .getResult(0);

        mlir::Value q8Group =
            (ib32 == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib32 * subBlock))
                      .getResult();

        for (int64_t l = 0; l < numGroups; ++l) {
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "grid_sign_group"));

          // int signs = sgn[ib32*4 + l];  (the EXPLICIT sign byte read DIRECTLY from
          // the signs region at xb+74 -- NO ksigns lookup).
          mlir::Value signs =
              loadByteAsInt(sgnBase, ib32 * signsPerSubBlock + l);

          // The two grid INDICES of this sign group, qh-injected (a SINGLE bit, mask
          // 256 = bit 8 -- NOT iq2_s's 2-bit 0x300):
          //   idx1 = qs[ib32*8 + 2l+0] | ((qhb << (8-2l)) & 256);  (pass A)
          //   idx2 = qs[ib32*8 + 2l+1] | ((qhb << (7-2l)) & 256);  (pass B)
          // The shifts DIFFER between the two passes (8-2l vs 7-2l). Both shift
          // amounts are positive for l in [0,3], so the int-domain shift is safe.
          mlir::Value qsByte1 =
              loadByteAsInt(qsBase, ib32 * indicesPerSubBlock + 2 * l + 0);
          mlir::Value qhShift1 =
              rewriter
                  .create<emitc::BitwiseLeftShiftOp>(loc, intType, qhByte,
                                                     intLit(8 - 2 * l))
                  .getResult();
          mlir::Value qhBit1 =
              rewriter
                  .create<emitc::BitwiseAndOp>(loc, intType, qhShift1,
                                               intLit(256))
                  .getResult();
          mlir::Value idx1 =
              rewriter
                  .create<emitc::BitwiseOrOp>(loc, intType, qsByte1, qhBit1)
                  .getResult();

          mlir::Value qsByte2 =
              loadByteAsInt(qsBase, ib32 * indicesPerSubBlock + 2 * l + 1);
          mlir::Value qhShift2 =
              rewriter
                  .create<emitc::BitwiseLeftShiftOp>(loc, intType, qhByte,
                                                     intLit(7 - 2 * l))
                  .getResult();
          mlir::Value qhBit2 =
              rewriter
                  .create<emitc::BitwiseAndOp>(loc, intType, qhShift2,
                                               intLit(256))
                  .getResult();
          mlir::Value idx2 =
              rewriter
                  .create<emitc::BitwiseOrOp>(loc, intType, qsByte2, qhBit2)
                  .getResult();

          // q8 pointers for the two 4-lane passes: pass A reads q8Group[0..3]
          // (kmask lanes 0..3), pass B reads q8Group[4..7] (kmask lanes 4..7).
          mlir::Value q8PassB =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group, sizeLit(4))
                  .getResult();

          // pass A: grid1 = grid_i8 + idx1*4 vs q8[0..3], kmaskLo {1,2,4,8}.
          sumiAcc = gridOf4Pass(idx1, signs, kmaskLo, q8Group, sumiAcc);
          // pass B: grid2 = grid_i8 + idx2*4 vs q8[4..7], kmaskHi {16,32,64,128}.
          sumiAcc = gridOf4Pass(idx2, signs, kmaskHi, q8PassB, sumiAcc);

          // q8Group += 8 (advance to the next group's 8 activations; each sign group
          // covers 8 elements = two grid-of-4 passes).
          int64_t groupElems = subBlock / numGroups; // 32/4 = 8
          q8Group =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                        sizeLit(groupElems))
                  .getResult();
        }

        // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (extract after the 4
        // groups / 8 passes -- the sub-block integer dot).
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, extractCallee));
        mlir::Value sumi =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                             extractCallee,
                                             mlir::ValueRange{sumiAcc})
                .getResult(0);

        // bsum = bsum + sumi * ls;  (integer accumulation; order-free).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, ls).getResult();
        mlir::Value sumiLs =
            rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32).getResult();
        mlir::Value bsumNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, bsumCur, sumiLs)
                .getResult();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("bsum", opName, role));
        rewriter.create<emitc::AssignOp>(loc, bsumVar, bsumNext);
      }

      // sumf = sumf + d * (float)bsum;  -- ONE emitc.expression so it renders as
      // ggml's single C statement and the compiler fuses the SAME FMA under
      // -ffp-contract=on/default. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value bsumFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value bsumFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, bsumFinal).getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, bsumFloat)
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

    // *s = sumf;  (iq3_s applies NO trailing factor -- a SEPARATE statement OUTSIDE
    // the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq3_s block-dot output not a pointer");
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

mlir::LogicalResult VariantToEmitCFunc::emitIQ2XSQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ2XSQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ2XSQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq2_xs block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq2_xs block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    // The 9-bit index `w & 511` and the sign selector `w >> 9` must be computed in
    // the UNSIGNED domain (ggml reads q2[l] as a uint16_t) so the >> is a LOGICAL
    // shift -- a signed `int` w with bit 15 set would arithmetic-shift and corrupt
    // the sign selector.
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                                  // 256
    int64_t subBlock = blockDot.getSubBlock();                      //  32
    int64_t weightStride = blockDot.getWeightBlockStride();         //  74
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   2
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();    //  66
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t groupLanes = 8;    // 8 grid values per group

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

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
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The GRID codebook is a STRUCTURAL fact off the typed attr (I4 mirror): the
    // 512 uint64 iq2xs_grid literals, emitted ONCE as a `static const int64_t[512]`
    // decl (rendering ggml's exact hex literals) and read as bytes through a
    // (const int8_t *) cast -- copying ggml's source method (`grid = (const uint8_t
    // *)(iq2xs_grid + idx)`). Every grid byte is <= 0x2b (43) < 128, so reading it
    // as int8 yields the identical numeric value as ggml's uint8 read.
    llvm::ArrayRef<int64_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const int64_t tcrv_iq2xs_grid[512] = {";
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

    // The SIGN PLANE is a STRUCTURAL fact off the typed attr (I4 mirror): the 128
    // ksigns_iq2xs entries (values reach 255, beyond int8, carried as i32 in the
    // attr) emitted ONCE as a `static const uint8_t[128]` decl, indexed by the
    // 7-bit sign selector.
    llvm::ArrayRef<int32_t> ksigns = blockDot.getKsigns();
    {
      std::string decl = "static const uint8_t tcrv_iq2xs_ksigns[128] = {";
      for (size_t i = 0; i < ksigns.size(); ++i) {
        if (i)
          decl += ", ";
        decl += std::to_string(static_cast<int>(ksigns[i]) & 0xff);
      }
      decl += "};";
      rewriter.create<emitc::VerbatimOp>(loc, decl);
    }

    // The kmask sign-bit selector {1<<j} is an inline const (NOT a table off the
    // attr -- it is a trivial bit-position vector), emitted as a decl + broadcast
    // load ONCE above the super-block loop.
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_iq2xs_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

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

    // vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_iq2xs_kmask, 8);  (ONCE)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "kmask_table_load"));
    mlir::Value kmaskName =
        rewriter.create<emitc::LiteralOp>(loc, u8PtrType, "tcrv_iq2xs_kmask");
    mlir::Value kmaskVec =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType},
                riscvIntrinsicName("vle", 8, coreLmul, "u8"),
                mlir::ValueRange{kmaskName, sizeLit(groupLanes)})
            .getResult(0);

    // const int8_t *grid_i8 = (const int8_t *)tcrv_iq2xs_grid;  (byte view of the
    // grid table; grid_i8 + idx*8 addresses the 8 int8 values of entry idx --
    // exactly ggml's `(const uint8_t *)(iq2xs_grid + idx)` byte address).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, "tcrv_iq2xs_grid");
    mlir::Value gridName =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrayName).getResult();

    // Per-super-block base address arithmetic: xb = vx + ibl*74; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The 9-bit index / sign-selector bitwise ops run in the UNSIGNED domain
    // (uint32_t) so the >> is a LOGICAL shift (ggml's q2[l] is uint16_t).
    auto uAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseAndOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseOrOp>(loc, uintType, a, b).getResult();
    };
    auto uShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseRightShiftOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uintType, a, b)
          .getResult();
    };
    auto uintLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, uintType,
                                               std::to_string(v) + "u");
    };
    // uint32_t x = (uint32_t)a[i];  -- a structured byte load from a `const uint8_t
    // *` then a cast to uint32_t (used to reassemble the uint16 weight word from the
    // 2-aligned qs stream alignment-safely; NO `*(uint16_t*)`).
    auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, uintType, u8).getResult();
    };
    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used to read the explicit scale byte sc[ib32] from the scales[]
    // stream alignment-safely).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
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

      // const uint8_t *qs = xb + 2;  const uint8_t *sc = xb + 66;  const int8_t *q8 = yb + 4;
      mlir::Value qsBase0 = xb;
      if (qsOffset != 0)
        qsBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qsOffset));
      mlir::Value qsBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBase0).getResult();
      mlir::Value scBase0 = xb;
      if (scalesOffset != 0)
        scBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(scalesOffset));
      mlir::Value scBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, scBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();

      // int32_t bsum = 0;  (the integer super-block accumulator, reset per
      // super-block; ggml's per-super-block bsum).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("bsum", opName, role));
      auto bsumVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, bsumVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib32 = 0..7), fully unrolled. Each sub-block
      // has 4 uint16 qs words (q2[0..3]) at qs + ib32*8 bytes and one scale byte
      // sc[ib32]; the q8 cursor advances 8 per group continuously.
      for (int64_t ib32 = 0; ib32 < numSubBlocks; ++ib32) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_explicit_scales"));
        // const uint8_t *a = qs + ib32*8;  (the 4 uint16 words = 8 bytes for this
        // sub-block).
        mlir::Value aBase =
            (ib32 == 0)
                ? qsBase
                : rewriter
                      .create<emitc::AddOp>(loc, u8PtrType, qsBase,
                                            sizeLit(ib32 * 8))
                      .getResult();

        // DELTA(c): int sc = sc_base[ib32];  ls1 = 2*(sc & 0xf)+1;  ls2 = 2*(sc>>4)+1.
        mlir::Value scByte = loadByteAsInt(scBase, ib32);
        auto buildScale =
            [&](mlir::Value nibble) -> mlir::Value {
          // ls = 2*nibble + 1, in the int domain.
          return rewriter
              .create<emitc::AddOp>(
                  loc, intType,
                  rewriter
                      .create<emitc::MulOp>(loc, intType, nibble, intLit(2))
                      .getResult(),
                  intLit(1))
              .getResult();
        };
        mlir::Value scLow =
            rewriter
                .create<emitc::BitwiseAndOp>(loc, intType, scByte, intLit(15))
                .getResult();
        mlir::Value scHigh =
            rewriter
                .create<emitc::BitwiseRightShiftOp>(loc, intType, scByte,
                                                    intLit(4))
                .getResult();
        mlir::Value ls1 = buildScale(scLow);
        mlir::Value ls2 = buildScale(scHigh);

        // The continuous q8 cursor for this sub-block's 4 groups (advances across
        // the half boundary -- ggml does NOT reset q8 between the {0,1} and {2,3}
        // halves).
        mlir::Value q8Group =
            (ib32 == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib32 * subBlock))
                      .getResult();

        // Two halves: h=0 -> groups l=0,1 scaled by ls1; h=1 -> groups l=2,3 by ls2.
        for (int64_t h = 0; h < 2; ++h) {
          // int32_t sumi = 0;  (the chained i32m1 reduction seed for this half's 2
          // groups; integer add is order-free).
          std::string seedCallee =
              riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seedCallee));
          mlir::Value zeroSeed =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          mlir::Value sumiAcc =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                               seedCallee,
                                               mlir::ValueRange{zeroSeed,
                                                                sizeLit(1)})
                  .getResult(0);

          for (int64_t lInHalf = 0; lInHalf < 2; ++lInHalf) {
            int64_t l = 2 * h + lInHalf;  // group index 0..3 within the sub-block
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "grid_sign_group"));
            // DELTA(b): uint16_t w = a[2*l] | a[2*l+1]<<8;  (2 LE byte loads from the
            // 2-aligned qs stream -- alignment-safe, unsigned so the >> is logical).
            mlir::Value w = loadByteAsUint(aBase, 2 * l);
            w = uOr(w, uShl(loadByteAsUint(aBase, 2 * l + 1), uintLit(8)));
            // DELTA(a): int idx = w & 511;  (the 9-bit grid index into the 512-grid).
            mlir::Value idx =
                rewriter
                    .create<emitc::CastOp>(loc, intType, uAnd(w, uintLit(511)))
                    .getResult();
            // int signs = tcrv_iq2xs_ksigns[w >> 9];  (the 7-bit sign selector; the
            // shift is logical in the uint32_t domain; cast to int for the table
            // subscript).
            mlir::Value signSel =
                rewriter
                    .create<emitc::CastOp>(loc, intType, uShr(w, uintLit(9)))
                    .getResult();
            mlir::Value ksignsName = rewriter.create<emitc::LiteralOp>(
                loc, u8PtrType, "tcrv_iq2xs_ksigns");
            mlir::Value signsElem =
                rewriter
                    .create<emitc::SubscriptOp>(
                        loc,
                        llvm::cast<mlir::TypedValue<emitc::PointerType>>(
                            ksignsName),
                        signSel)
                    .getResult();
            mlir::Value signsU8 =
                rewriter.create<emitc::LoadOp>(loc, constU8Type, signsElem)
                    .getResult();
            mlir::Value signs =
                rewriter.create<emitc::CastOp>(loc, intType, signsU8).getResult();

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

            // const int8_t *grid = grid_i8 + idx*8;  (indexed pointer arith).
            mlir::Value idxLane =
                rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(8))
                    .getResult();
            mlir::Value gridPtr =
                rewriter.create<emitc::AddOp>(loc, i8PtrType, gridName, idxLane)
                    .getResult();
            // grid_v = vle8_v_i8m1(grid, vl);  q8_v = vle8_v_i8m1(q8Group, vl);
            std::string i8LoadCallee =
                riscvIntrinsicName("vle", 8, coreLmul, "i8");
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, i8LoadCallee));
            mlir::Value gridV =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                                 i8LoadCallee,
                                                 mlir::ValueRange{gridPtr, vl})
                    .getResult(0);
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, i8LoadCallee));
            mlir::Value q8V =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                                 i8LoadCallee,
                                                 mlir::ValueRange{q8Group, vl})
                    .getResult(0);

            // sign-bit mask: m = vmsne(vand(vmv(signs), kmask), 0).
            std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, bcastCallee));
            mlir::Value signsBcast =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                                 bcastCallee,
                                                 mlir::ValueRange{signs, vl})
                    .getResult(0);
            std::string andCallee = "__riscv_vand_vv_u8m1";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, andCallee));
            mlir::Value signBits =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{u8CoreType}, andCallee,
                        mlir::ValueRange{signsBcast, kmaskVec, vl})
                    .getResult(0);
            std::string msneCallee =
                riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, msneCallee));
            mlir::Value signMask =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{maskType}, msneCallee,
                        mlir::ValueRange{signBits, intLit(0), vl})
                    .getResult(0);

            // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign).
            std::string negCallee = "__riscv_vneg_v_i8m1";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, negCallee));
            mlir::Value gridNeg =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                                 negCallee,
                                                 mlir::ValueRange{gridV, vl})
                    .getResult(0);
            std::string mergeCallee =
                riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, mergeCallee));
            mlir::Value gridSigned =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{i8CoreType}, mergeCallee,
                        mlir::ValueRange{gridV, gridNeg, signMask, vl})
                    .getResult(0);

            // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
            // <= 43*127 = 5461 < 32767, fits i16).
            std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, wmulCallee));
            mlir::Value product =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{i16WideType}, wmulCallee,
                        mlir::ValueRange{gridSigned, q8V, vl})
                    .getResult(0);

            // sumiAcc = vwredsum(product, sumiAcc, vl);  (chain the i32 reduction;
            // integer add is order-free, so seeding from the running accumulator is
            // byte-exact across the 2 groups of this half).
            std::string reduceCallee =
                ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, reduceCallee));
            sumiAcc =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                        mlir::ValueRange{product, sumiAcc, vl})
                    .getResult(0);

            // q8Group += 8 (advance to the next group's 8 activations; continuous
            // across the half boundary).
            q8Group =
                rewriter
                    .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                          sizeLit(groupLanes))
                    .getResult();
          }

          // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (extract after the 2
          // groups of this half -- the half's integer dot).
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extractCallee));
          mlir::Value sumi =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                               extractCallee,
                                               mlir::ValueRange{sumiAcc})
                  .getResult(0);

          // bsum = bsum + sumi * (h ? ls2 : ls1);  (integer accumulation;
          // order-free; DELTA(c): the explicit per-half scale).
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "bsum_accumulate"));
          mlir::Value bsumCur =
              rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
          mlir::Value lsI32 =
              rewriter.create<emitc::CastOp>(loc, i32Type, (h == 0) ? ls1 : ls2)
                  .getResult();
          mlir::Value sumiLs =
              rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32)
                  .getResult();
          mlir::Value bsumNext =
              rewriter.create<emitc::AddOp>(loc, i32Type, bsumCur, sumiLs)
                  .getResult();
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("bsum", opName, role));
          rewriter.create<emitc::AssignOp>(loc, bsumVar, bsumNext);
        }
      }

      // sumf = sumf + d * (float)bsum;  -- ONE emitc.expression so it renders as
      // ggml's single C statement and the compiler fuses the SAME FMA under
      // -ffp-contract=on/default. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value bsumFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value bsumFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, bsumFinal).getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, bsumFloat)
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

    // *s = 0.125f * sumf;  (the iq2_xs trailing 1/8 factor, a SEPARATE statement
    // OUTSIDE the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq2_xs block-dot output not a pointer");
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

    valueMap[blockDot.getResult()] = scaled;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitIQ2SQ8KBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotIQ2SQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotIQ2SQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "iq2_s block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq2_s block-dot ABI operand unmapped");

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
    int64_t weightStride = blockDot.getWeightBlockStride();         //  82
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();        //   0
    int64_t qsOffset = blockDot.getWeightQsByteOffset();            //   2
    int64_t signsOffset = blockDot.getWeightSignsByteOffset();      //  34
    int64_t qhOffset = blockDot.getWeightQhByteOffset();            //  66
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();    //  74
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();     //   4
    int64_t numSubBlocks = qk / subBlock;                           //   8
    int64_t groupsPerSub = 4;  // 4 grid groups per sub-block (l=0..3)
    int64_t groupLanes = 8;    // 8 grid values per group

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

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
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The GRID codebook is a STRUCTURAL fact off the typed attr (I4 mirror): the
    // 1024 uint64 iq2s_grid literals, emitted ONCE as a `static const int64_t[1024]`
    // decl (rendering ggml's exact hex literals) and read as bytes through a
    // (const int8_t *) cast -- copying ggml's source method (`grid = (const uint8_t
    // *)(iq2s_grid + idx)`). Every grid byte is <= 0x2b (43) < 128, so reading it
    // as int8 yields the identical numeric value as ggml's uint8 read.
    llvm::ArrayRef<int64_t> grid = blockDot.getGrid();
    {
      std::string decl = "static const int64_t tcrv_iq2s_grid[1024] = {";
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

    // The kmask sign-bit selector {1<<j} is an inline const (a trivial bit-position
    // vector), emitted as a decl + broadcast load ONCE above the super-block loop.
    // iq2_s has NO ksigns plane (the signs are an explicit memory region).
    rewriter.create<emitc::VerbatimOp>(
        loc, "static const uint8_t tcrv_iq2s_kmask[8] = {1, 2, 4, 8, 16, 32, "
             "64, 128};");

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

    // vuint8m1_t kmask = __riscv_vle8_v_u8m1(tcrv_iq2s_kmask, 8);  (ONCE)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "kmask_table_load"));
    mlir::Value kmaskName =
        rewriter.create<emitc::LiteralOp>(loc, u8PtrType, "tcrv_iq2s_kmask");
    mlir::Value kmaskVec =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{u8CoreType},
                riscvIntrinsicName("vle", 8, coreLmul, "u8"),
                mlir::ValueRange{kmaskName, sizeLit(groupLanes)})
            .getResult(0);

    // const int8_t *grid_i8 = (const int8_t *)tcrv_iq2s_grid;  (byte view of the
    // grid table; grid_i8 + idx*8 addresses the 8 int8 values of entry idx --
    // exactly ggml's `(const uint8_t *)(iq2s_grid + idx)` byte address).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "grid_table_byte_view"));
    mlir::Type i64PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int64_t"));
    mlir::Value gridArrayName =
        rewriter.create<emitc::LiteralOp>(loc, i64PtrType, "tcrv_iq2s_grid");
    mlir::Value gridName =
        rewriter.create<emitc::CastOp>(loc, i8PtrType, gridArrayName).getResult();

    // Per-super-block base address arithmetic: xb = vx + ibl*82; yb = vy + ibl*292.
    auto blockBaseValue = [&](mlir::Value ibl, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ibl, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used to read the single qs index byte, the explicit sign byte,
    // the qh-plane byte, and the explicit scale byte alignment-safely).
    auto loadByteAsInt = [&](mlir::Value ptr, int64_t i) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(i));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(ptr), idx)
              .getResult();
      mlir::Value u8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, elem).getResult();
      return rewriter.create<emitc::CastOp>(loc, intType, u8).getResult();
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

      // const uint8_t *qs = xb + 2;  const uint8_t *sgn = xb + 34;
      // const uint8_t *qh = xb + 66;  const uint8_t *sc = xb + 74;
      // const int8_t *q8 = yb + 4;
      mlir::Value qsBase0 = xb;
      if (qsOffset != 0)
        qsBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                sizeLit(qsOffset));
      mlir::Value qsBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, qsBase0).getResult();
      mlir::Value sgnBase0 = xb;
      if (signsOffset != 0)
        sgnBase0 = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                                 sizeLit(signsOffset));
      mlir::Value sgnBase =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, sgnBase0).getResult();
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
          rewriter.create<emitc::CastOp>(loc, u8PtrType, scBase0).getResult();
      mlir::Value q8Base0 = yb;
      if (q8Offset != 0)
        q8Base0 = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                sizeLit(q8Offset));
      mlir::Value q8Base =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Base0).getResult();

      // int32_t bsum = 0;  (the integer super-block accumulator, reset per
      // super-block; ggml's per-super-block bsum).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("bsum", opName, role));
      auto bsumVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, bsumVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The FLAT per-sub-block loop (ib32 = 0..7), fully unrolled. Each sub-block
      // has 4 single-byte qs index bytes (qs[ib32*4 + l], l=0..3), 4 explicit sign
      // bytes (sgn[ib32*4 + l]), one qh-plane byte (qh[ib32]) and one scale byte
      // sc[ib32]; the q8 cursor advances 8 per group continuously.
      for (int64_t ib32 = 0; ib32 < numSubBlocks; ++ib32) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_explicit_scales"));

        // int sc = sc_base[ib32];  ls1 = 2*(sc & 0xf)+1;  ls2 = 2*(sc>>4)+1.
        mlir::Value scByte = loadByteAsInt(scBase, ib32);
        auto buildScale =
            [&](mlir::Value nibble) -> mlir::Value {
          // ls = 2*nibble + 1, in the int domain.
          return rewriter
              .create<emitc::AddOp>(
                  loc, intType,
                  rewriter
                      .create<emitc::MulOp>(loc, intType, nibble, intLit(2))
                      .getResult(),
                  intLit(1))
              .getResult();
        };
        mlir::Value scLow =
            rewriter
                .create<emitc::BitwiseAndOp>(loc, intType, scByte, intLit(15))
                .getResult();
        mlir::Value scHigh =
            rewriter
                .create<emitc::BitwiseRightShiftOp>(loc, intType, scByte,
                                                    intLit(4))
                .getResult();
        mlir::Value ls1 = buildScale(scLow);
        mlir::Value ls2 = buildScale(scHigh);

        // DELTA(b): int qhb = qh[ib32];  (the per-sub-block qh-bit plane byte; its
        // 2-bit fields inject the high bits of each group's 10-bit grid index).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_plane_byte"));
        mlir::Value qhByte = loadByteAsInt(qhBase, ib32);

        // The continuous q8 cursor for this sub-block's 4 groups (advances across
        // the half boundary -- ggml does NOT reset q8 between the {0,1} and {2,3}
        // halves).
        mlir::Value q8Group =
            (ib32 == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib32 * subBlock))
                      .getResult();

        // Two halves: h=0 -> groups l=0,1 scaled by ls1; h=1 -> groups l=2,3 by ls2.
        for (int64_t h = 0; h < 2; ++h) {
          // int32_t sumi = 0;  (the chained i32m1 reduction seed for this half's 2
          // groups; integer add is order-free).
          std::string seedCallee =
              riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seedCallee));
          mlir::Value zeroSeed =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          mlir::Value sumiAcc =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                               seedCallee,
                                               mlir::ValueRange{zeroSeed,
                                                                sizeLit(1)})
                  .getResult(0);

          for (int64_t lInHalf = 0; lInHalf < 2; ++lInHalf) {
            int64_t l = 2 * h + lInHalf;  // group index 0..3 within the sub-block
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "grid_sign_group"));
            // DELTA(a)+(b): idx = qs[ib32*4 + l] | ((qhb << (8-2*l)) & 0x300).
            // The low 8 bits are a single qs index byte; the high 2 bits are the
            // (8-2*l)-shifted qh-plane field masked to bits [8,9]. shift = 8,6,4,2
            // for l = 0,1,2,3. The whole index is computed in the int domain
            // (`qs[]`/`qh[]` are uint8 and the shift amount is small/positive, so no
            // sign-extension hazard; the result stays within [0,1023]).
            mlir::Value qsIdxByte =
                loadByteAsInt(qsBase, ib32 * groupsPerSub + l);
            mlir::Value qhShifted =
                rewriter
                    .create<emitc::BitwiseLeftShiftOp>(loc, intType, qhByte,
                                                       intLit(8 - 2 * l))
                    .getResult();
            mlir::Value qhHighBits =
                rewriter
                    .create<emitc::BitwiseAndOp>(loc, intType, qhShifted,
                                                 intLit(0x300))
                    .getResult();
            mlir::Value idx =
                rewriter
                    .create<emitc::BitwiseOrOp>(loc, intType, qsIdxByte,
                                                qhHighBits)
                    .getResult();
            // DELTA(c): int signs = sgn[ib32*4 + l];  (the EXPLICIT sign byte read
            // DIRECTLY from the sign region at qs+32 -- NO ksigns lookup).
            mlir::Value signs =
                loadByteAsInt(sgnBase, ib32 * groupsPerSub + l);

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

            // const int8_t *grid = grid_i8 + idx*8;  (indexed pointer arith).
            mlir::Value idxLane =
                rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(8))
                    .getResult();
            mlir::Value gridPtr =
                rewriter.create<emitc::AddOp>(loc, i8PtrType, gridName, idxLane)
                    .getResult();
            // grid_v = vle8_v_i8m1(grid, vl);  q8_v = vle8_v_i8m1(q8Group, vl);
            std::string i8LoadCallee =
                riscvIntrinsicName("vle", 8, coreLmul, "i8");
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, i8LoadCallee));
            mlir::Value gridV =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                                 i8LoadCallee,
                                                 mlir::ValueRange{gridPtr, vl})
                    .getResult(0);
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, i8LoadCallee));
            mlir::Value q8V =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                                 i8LoadCallee,
                                                 mlir::ValueRange{q8Group, vl})
                    .getResult(0);

            // sign-bit mask: m = vmsne(vand(vmv(signs), kmask), 0).
            std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, bcastCallee));
            mlir::Value signsBcast =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                                 bcastCallee,
                                                 mlir::ValueRange{signs, vl})
                    .getResult(0);
            std::string andCallee = "__riscv_vand_vv_u8m1";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, andCallee));
            mlir::Value signBits =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{u8CoreType}, andCallee,
                        mlir::ValueRange{signsBcast, kmaskVec, vl})
                    .getResult(0);
            std::string msneCallee =
                riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, msneCallee));
            mlir::Value signMask =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{maskType}, msneCallee,
                        mlir::ValueRange{signBits, intLit(0), vl})
                    .getResult(0);

            // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign).
            std::string negCallee = "__riscv_vneg_v_i8m1";
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, negCallee));
            mlir::Value gridNeg =
                rewriter
                    .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                                 negCallee,
                                                 mlir::ValueRange{gridV, vl})
                    .getResult(0);
            std::string mergeCallee =
                riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, mergeCallee));
            mlir::Value gridSigned =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{i8CoreType}, mergeCallee,
                        mlir::ValueRange{gridV, gridNeg, signMask, vl})
                    .getResult(0);

            // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
            // <= 43*127 = 5461 < 32767, fits i16).
            std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, wmulCallee));
            mlir::Value product =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{i16WideType}, wmulCallee,
                        mlir::ValueRange{gridSigned, q8V, vl})
                    .getResult(0);

            // sumiAcc = vwredsum(product, sumiAcc, vl);  (chain the i32 reduction;
            // integer add is order-free, so seeding from the running accumulator is
            // byte-exact across the 2 groups of this half).
            std::string reduceCallee =
                ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, reduceCallee));
            sumiAcc =
                rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                        mlir::ValueRange{product, sumiAcc, vl})
                    .getResult(0);

            // q8Group += 8 (advance to the next group's 8 activations; continuous
            // across the half boundary).
            q8Group =
                rewriter
                    .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                          sizeLit(groupLanes))
                    .getResult();
          }

          // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (extract after the 2
          // groups of this half -- the half's integer dot).
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extractCallee));
          mlir::Value sumi =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                               extractCallee,
                                               mlir::ValueRange{sumiAcc})
                  .getResult(0);

          // bsum = bsum + sumi * (h ? ls2 : ls1);  (integer accumulation;
          // order-free; the explicit per-half scale).
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "bsum_accumulate"));
          mlir::Value bsumCur =
              rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
          mlir::Value lsI32 =
              rewriter.create<emitc::CastOp>(loc, i32Type, (h == 0) ? ls1 : ls2)
                  .getResult();
          mlir::Value sumiLs =
              rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32)
                  .getResult();
          mlir::Value bsumNext =
              rewriter.create<emitc::AddOp>(loc, i32Type, bsumCur, sumiLs)
                  .getResult();
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("bsum", opName, role));
          rewriter.create<emitc::AssignOp>(loc, bsumVar, bsumNext);
        }
      }

      // sumf = sumf + d * (float)bsum;  -- ONE emitc.expression so it renders as
      // ggml's single C statement and the compiler fuses the SAME FMA under
      // -ffp-contract=on/default. Invoked in STRICT ascending super-block order.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value bsumFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value bsumFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, bsumFinal).getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d, bsumFloat)
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

    // *s = 0.125f * sumf;  (the iq2_s trailing 1/8 factor, a SEPARATE statement
    // OUTSIDE the accumulate expression; structured scalar store through *s).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "iq2_s block-dot output not a pointer");
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

    valueMap[blockDot.getResult()] = scaled;
    return mlir::success();
  }

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
