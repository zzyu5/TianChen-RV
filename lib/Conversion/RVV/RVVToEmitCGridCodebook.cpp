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

#include <optional>
#include <string>
#include <utility>

namespace weft {
namespace conversion {
namespace rvv {
namespace detail {

// VariantToEmitCFunc grid-codebook emit methods: iq2_xxs/iq2_xs/iq2_s and
// iq3_xxs/iq3_s (grid + sign-plane decode). Split out of RVVToEmitC.cpp as a
// pure code move; the emitted C is byte-identical.

// M-FLAT iq2_xxs super-block GRID-of-8 byte-exact SHARED body anchor. Extracted from
// the (now-retired) monolith emitIQ2XXSQ8KBlockDot as a pure code move (the emitted C
// is byte-identical) so the front-door-constructed typed super-block SCALAR-grid loop
// (fold_model "scalar_delta_grid", stride 66) lowers byte-identically by construction:
// same per-super-block body, same facts, same order. It emits ONE super-block's body
// at the current insertion point INSIDE an already-open super-block loop whose
// per-super-block bases xb/yb are provided (the grid/signs64 decls, the sumf/nb setup,
// the ONCE grid64 + signs64 i64 views, the outer loop, and the trailing `*s =
// 0.125f*sumf` store live in the wrapper). The emitc element/pointer types + the load
// helpers are re-derived here from the MLIRContext (uniqued -> the SAME Type instances)
// so the emit is byte-identical to the monolith's inline body. The coreLmul is the
// Win-A gearbox anchor (m2 default / m1 at VLEN256) carried on the grid-core brick; the
// dot wide LMUL is 2*core (i16m4 at m2 / i16m2 at m1); the u16 index EMUL = (16/64)*core
// (mf2 at m2 / mf4 at m1). The whole body is byte-exact for any legal anchor.
void VariantToEmitCFunc::emitIQ2XXSSuperBlockGridBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const IQ2XXSGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    int64_t weightDOffset = cx.weightDOffset;          //   0
    int64_t qsOffset = cx.qsOffset;                    //   2
    int64_t activationDOffset = cx.activationDOffset;  //   0
    int64_t q8Offset = cx.q8Offset;                    //   4
    int64_t numSubBlocks = cx.numSubBlocks;            //   8
    int64_t numGroups = cx.numGroups;                  //   4
    llvm::StringRef coreLmul = cx.coreLmul;
    mlir::Value gridName = cx.gridName;
    mlir::Value signs64 = cx.signs64;

    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    // aux1 + the scale `ls = 2*(aux1>>28)+1` + the sign selector `(aux1>>7*l)&127` must
    // be computed in the UNSIGNED domain (ggml reads aux32[1] as a uint32_t) so the >>
    // is a LOGICAL shift -- a signed `int` aux1 with bit 31 set would arithmetic-shift
    // and corrupt the scale/selector.
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");

    // The per-sub-block dot wide LMUL is 2*core (FLIPS with the anchor, NOT always m4).
    // The batched gather widths (2*core + its u16 index EMUL) are derived per-pair below.
    llvm::StringRef wideLmul = (coreLmul == "m2") ? "m4" : "m2";
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    int64_t subBlockLanes = 32; // 4 grid entries * 8 i8 = one 32-lane sub-block
    mlir::Type i8WideType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i16WidestType =
        emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // The aux1 / scale / sign-selector bitwise ops run in the UNSIGNED domain (uint32_t)
    // so the >> is a LOGICAL shift (ggml's aux32[1] is uint32_t).
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
    auto uintLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
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
    // int x = (int)a[i];  -- alignment-safe byte load then cast to int, emitted by
    // the shared emitLoadByteAsInt(constU8Type, intType) helper.

    // d = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);  (ONCE per
    // super-block; the fp16 weight scale times the fp32 q8_K scale).
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

    // int32_t bsum = 0;  (the integer super-block accumulator, reset per super-block;
    // ggml's per-super-block bsum).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("bsum", opName, role));
    auto bsumVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, bsumVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

    // The per-sub-block dot is BATCHED into PAIRS (pair = 0..numSubBlocks/2-1): each
    // pair decodes TWO sub-blocks (s0 = 2*pair, s1 = 2*pair+1) with ONE wider
    // i64<2*core> grid gather + ONE wider signs64 gather (8 index slots), ONE wider
    // i8<2*core> q8 load, and ONE wider i8<2*core> vmul sign-fold. The per-sub-block
    // signed widening product + vwredsum are then recovered from each 32-lane half via a
    // register-group vget (i8<2*core> -> i8<core>), so the integer dot per sub-block is
    // BYTE-IDENTICAL to the unbatched body while the gather + vsetvli config are hoisted
    // to the super-block pair (16 gathers -> 8, halved vle16/vle8/vmul + config churn).
    // bsum still accumulates in STRICT ascending sub-block order (s0 then s1), each with
    // its own `ls`; the vwmul/vwredsum stay at the Win-A wide<->core LMUL gearbox widths.
    int64_t pairLanes = 2 * subBlockLanes;     // 64 = two 32-lane sub-blocks
    int64_t numGroupsPair = 2 * numGroups;     // 8 i64 grid/sign entries per pair
    // 2*core (the wide gather/load/fold LMUL) and its u16 index EMUL (16/64)*(2*core).
    llvm::StringRef pairLmul = (coreLmul == "m2") ? "m4" : "m2";
    llvm::StringRef pairIdxLmul = (coreLmul == "m2") ? "m1" : "mf2";
    mlir::Type i8PairType =
        emitc::OpaqueType::get(ctx, ("vint8" + pairLmul + "_t").str());
    mlir::Type i64PairType =
        emitc::OpaqueType::get(ctx, ("vint64" + pairLmul + "_t").str());
    mlir::Type u16IdxPairType =
        emitc::OpaqueType::get(ctx, ("vuint16" + pairIdxLmul + "_t").str());
    mlir::Type idxArrayPairType = emitc::ArrayType::get({8}, u16ElemType);
    std::string vgetCallee =
        ("__riscv_vget_v_i8" + pairLmul + "_i8" + coreLmul).str();

    for (int64_t pair = 0; pair < numSubBlocks / 2; ++pair) {
      // Decode BOTH sub-blocks of the pair up front: the per-sub-block scale ls[half]
      // and the 8-slot grid/sign byte-offset index arrays (slots [half*4 + l]). These
      // are pure integer decode ops -- reordering the two halves' decode changes no
      // value; the gather reads the fully-filled arrays.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "sub_block_aux_scale"));
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("gridoff", opName, role));
      auto gridOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto gridOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(gridOffVar.getResult());
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("signoff", opName, role));
      auto signOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto signOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(signOffVar.getResult());
      auto storeU16 = [&](mlir::TypedValue<emitc::ArrayType> arr, int64_t slot,
                          mlir::Value byteOff) {
        mlir::Value byteOffU16 =
            rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff).getResult();
        mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(slot));
        mlir::Value slotElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{slotIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
            byteOffU16);
      };

      mlir::Value lsPair[2];
      for (int64_t half = 0; half < 2; ++half) {
        int64_t ib32 = 2 * pair + half;
        // const uint8_t *a = qs + ib32*8;  (the 8 aux bytes for this sub-block).
        mlir::Value aBase =
            (ib32 == 0)
                ? qsBase
                : rewriter
                      .create<emitc::AddOp>(loc, u8PtrType, qsBase,
                                            sizeLit(ib32 * 8))
                      .getResult();
        // uint32_t aux1 = a[4] | a[5]<<8 | a[6]<<16 | a[7]<<24;  (little-endian, uint
        // domain -> logical >>).
        mlir::Value aux1 = loadByteAsUint(aBase, 4);
        aux1 = uOr(aux1, uShl(loadByteAsUint(aBase, 5), uintLit(8)));
        aux1 = uOr(aux1, uShl(loadByteAsUint(aBase, 6), uintLit(16)));
        aux1 = uOr(aux1, uShl(loadByteAsUint(aBase, 7), uintLit(24)));
        // int ls = 2*(aux1 >> 28) + 1;  (the per-sub-block 4-bit scale, [1,31]).
        mlir::Value lsTop =
            rewriter.create<emitc::CastOp>(loc, intType, uShr(aux1, uintLit(28)))
                .getResult();
        lsPair[half] =
            rewriter
                .create<emitc::AddOp>(
                    loc, intType,
                    rewriter.create<emitc::MulOp>(loc, intType, lsTop, intLit(2))
                        .getResult(),
                    intLit(1))
                .getResult();
        // Fill slots [half*4 + l]: 4 grid byte-offsets a[l]*8 + 4 sign byte-offsets
        // ((aux1>>7l)&127)*8 (the shift is logical in the uint32 domain).
        for (int64_t l = 0; l < numGroups; ++l) {
          mlir::Value idx = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, aBase, l);
          mlir::Value gridByteOff =
              rewriter.create<emitc::MulOp>(loc, intType, idx, intLit(8))
                  .getResult();
          storeU16(gridOffArray, half * numGroups + l, gridByteOff);
          mlir::Value sel =
              rewriter
                  .create<emitc::CastOp>(
                      loc, intType,
                      uAnd(uShr(aux1, uintLit(7 * l)), uintLit(127)))
                  .getResult();
          mlir::Value signByteOff =
              rewriter.create<emitc::MulOp>(loc, intType, sel, intLit(8))
                  .getResult();
          storeU16(signOffArray, half * numGroups + l, signByteOff);
        }
      }

      // The BATCHED vluxei16 IQ-gather: ONE wider gather over the whole PAIR. vle16 the
      // 8 u16 grid/sign indices ((2*core) EMUL), TWO __riscv_vluxei16_v_i64<2*core>
      // gathers over grid64 + signs64, each reinterpreted to i8<2*core> (64 grid bytes /
      // 64 +-1 sign bytes), ONE wider vle8_v_i8<2*core> q8 pair load + ONE vmul-onto-grid
      // sign fold. The gather/config are hoisted to the pair; the per-sub-block dot below
      // is byte-identical.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "grid_sign_subblock"));
      auto loadIdxPair =
          [&](mlir::TypedValue<emitc::ArrayType> arr) -> mlir::Value {
        mlir::Value base0 =
            rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
        mlir::Value baseElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{base0})
                .getResult();
        mlir::Value basePtr =
            rewriter.create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", baseElem)
                .getResult();
        std::string idxLoadCallee =
            riscvIntrinsicName("vle", 16, pairIdxLmul, "u16");
        return emitOpaqueCallBuilt(
            rewriter, loc, u16IdxPairType, idxLoadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {basePtr, sizeLit(numGroupsPair)};
            });
      };
      mlir::Value vGridOff = loadIdxPair(gridOffArray);
      mlir::Value vSignOff = loadIdxPair(signOffArray);

      std::string gridGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", pairLmul);
      mlir::Value gridGathered = emitOpaqueCallBuilt(
          rewriter, loc, i64PairType, gridGatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridName, vGridOff, sizeLit(numGroupsPair)};
          });
      std::string gridReinterpretCallee =
          ("__riscv_vreinterpret_v_i64" + pairLmul + "_i8" + pairLmul).str();
      mlir::Value gridPairV = emitOpaqueCall(rewriter, loc, i8PairType,
                                             gridReinterpretCallee,
                                             mlir::ValueRange{gridGathered},
                                             opName, role);

      std::string signGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", pairLmul);
      mlir::Value signGathered = emitOpaqueCallBuilt(
          rewriter, loc, i64PairType, signGatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signs64, vSignOff, sizeLit(numGroupsPair)};
          });
      std::string signReinterpretCallee =
          ("__riscv_vreinterpret_v_i64" + pairLmul + "_i8" + pairLmul).str();
      mlir::Value signsPairV = emitOpaqueCall(rewriter, loc, i8PairType,
                                              signReinterpretCallee,
                                              mlir::ValueRange{signGathered},
                                              opName, role);

      // const int8_t *q8pair = q8 + pair*64;  vint8<2*core> q8v = vle8(q8pair, 64).
      mlir::Value q8PairBase =
          (pair == 0)
              ? q8Base
              : rewriter
                    .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                          sizeLit(pair * pairLanes))
                    .getResult();
      std::string i8PairLoadCallee =
          riscvIntrinsicName("vle", 8, pairLmul, "i8");
      mlir::Value q8PairV = emitOpaqueCallBuilt(
          rewriter, loc, i8PairType, i8PairLoadCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {q8PairBase, sizeLit(pairLanes)};
          });

      // vint8<2*core> gs = vmul_vv_i8<2*core>(grid, signs, 64) -- sign folded onto the
      // GRID (not q8; q8 can be -128 and vmul(-128,-1) wraps).
      std::string signMulCallee = ("__riscv_vmul_vv_i8" + pairLmul).str();
      mlir::Value gridSignedPair = emitOpaqueCallBuilt(
          rewriter, loc, i8PairType, signMulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridPairV, signsPairV, sizeLit(pairLanes)};
          });

      // Per-sub-block dot: recover each 32-lane half via a register-group vget, then the
      // SAME i8<core> -> i16<wide> vwmul + vwredsum + extract as the unbatched body, in
      // STRICT s0-then-s1 order with each half's own `ls`.
      for (int64_t half = 0; half < 2; ++half) {
        mlir::Value gridSigned = emitOpaqueCall(
            rewriter, loc, i8WideType, vgetCallee,
            mlir::ValueRange{gridSignedPair, sizeLit(half)}, opName, role);
        mlir::Value q8V = emitOpaqueCall(rewriter, loc, i8WideType, vgetCallee,
                                         mlir::ValueRange{q8PairV, sizeLit(half)},
                                         opName, role);

        // p = vwmul_vv_i16<wide>(gridSigned, q8v, 32);  (each lane <= 43*127 < 32767).
        std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
        mlir::Value product = emitOpaqueCallBuilt(
            rewriter, loc, i16WidestType, wmulCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {gridSigned, q8V, sizeLit(subBlockLanes)};
            });
        // int32_t sumi = vmv_x_s(vwredsum(p, vmv_v_x(0,1), 32));  ONE reduction/sub-block.
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroSeed =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zeroSeed, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, reduceCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {product, sumiAcc, sizeLit(subBlockLanes)};
            });
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value sumi =
            emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                           mlir::ValueRange{sumiAcc}, opName, role);

        // bsum = bsum + sumi * ls;  (integer accumulation; strict s0-then-s1 order).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, lsPair[half])
                .getResult();
        mlir::Value sumiLs =
            rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32).getResult();
        mlir::Value bsumNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, bsumCur, sumiLs)
                .getResult();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("bsum", opName, role));
        rewriter.create<emitc::AssignOp>(loc, bsumVar, bsumNext);
      }
    }

    // sumf = sumf + d * (float)bsum;  -- ONE emitc.expression so it renders as ggml's
    // single C statement and the compiler fuses the SAME FMA under -ffp-contract on/
    // default. Invoked in STRICT ascending super-block order.
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

// M-FLAT iq3_xxs super-block GRID-of-4 byte-exact SHARED body anchor. Extracted from
// the (now-retired) monolith emitIQ3XXSQ8KBlockDot as a pure code move (the emitted C
// is byte-identical) so the front-door-constructed typed super-block SCALAR-grid loop
// (fold_model "scalar_delta_grid", stride 98) lowers byte-identically by construction:
// same per-super-block body, same facts, same order. It emits ONE super-block's body
// at the current insertion point INSIDE an already-open super-block loop whose
// per-super-block bases xb/yb are provided (the grid/ksigns/kmask decls, the sumf/nb
// setup, the ONCE 8-lane kmask load + grid32 view, the outer loop, and the trailing
// `*s = 0.25f*sumf` store live in the wrapper). The emitc element/pointer types + the
// load helpers are re-derived here from the MLIRContext (uniqued -> the SAME Type
// instances) so the emit is byte-identical to the monolith's inline body.
void VariantToEmitCFunc::emitIQ3XXSSuperBlockGridBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const IQ3XXSGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    int64_t weightDOffset = cx.weightDOffset;          //   0
    int64_t qsOffset = cx.qsOffset;                    //   2
    int64_t gasOffset = cx.gasOffset;                  //  66
    int64_t activationDOffset = cx.activationDOffset;  //   0
    int64_t q8Offset = cx.q8Offset;                    //   4
    int64_t subBlock = cx.subBlock;                    //  32
    int64_t numSubBlocks = cx.numSubBlocks;            //   8
    int64_t numGroups = cx.numGroups;                  //   4
    int64_t indicesPerSubBlock = cx.indicesPerSubBlock;//   8
    mlir::Value grid32 = cx.grid32;
    mlir::Value kmask = cx.kmask;

    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    // aux32 + the scale `ls = 2*(aux32>>28)+1` + the sign selector `(aux32>>7*l)&127`
    // must be computed in the UNSIGNED domain (ggml reads aux32 as a uint32_t) so the
    // >> is a LOGICAL shift -- a signed `int` aux32 with bit 31 set would arithmetic-
    // shift and corrupt the scale/selector (the iq2_xxs hardware-bisected bug).
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    int64_t groupLanes = 8;    // 8 grid lanes per sign group (the two old 4-lane passes)
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

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
    auto uintLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
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
    // Emitted by the shared emitLoadByteAsInt(constU8Type, intType) helper.

    // ONE grid-of-4 DOT: the per-group sign-fold + signed widening dot, fed a
    // PRE-GATHERED gridV (this group's 8 signed grid bytes in lanes 0..7, recovered by a
    // register-group vget from the SUPER-BLOCK-PAIR-batched vluxei16 gather emitted in the
    // pair loop below). The sign-fold ops (vmv/vand/vmsne/vneg/vmerge), the widening
    // product, and the chained vwredsum are BYTE-IDENTICAL to the unbatched body -- only
    // the per-group vl=2 __riscv_vluxei16_v_i32m1 grid gather (the fractional-LMUL
    // 2-element scalarization) is hoisted out. Lane mapping is UNCHANGED: lanes 0..3 =
    // grid1/q8[0..3]/kmask{1,2,4,8} (old pass A), lanes 4..7 = grid2/q8[4..7]/kmask{16,32,
    // 64,128} (old pass B), so the per-lane product is byte-identical; the i32 reduction
    // is order-free.
    auto gridOf4Dot = [&](mlir::Value gridV, mlir::Value signs, mlir::Value q8Ptr,
                          mlir::Value sumiAcc) -> mlir::Value {

      // vint8m1_t q8v = __riscv_vle8_v_i8m1(q8Ptr, 8);  (the 8 group activations.)
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      mlir::Value q8V = emitOpaqueCallBuilt(
          rewriter, loc, i8CoreType, i8LoadCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {q8Ptr, sizeLit(groupLanes)};
          });

      // sign-bit mask: m = vmsne(vand(vmv(signs), kmask), 0).  (UNCHANGED ops, vl=8,
      // full 8-bit kmask -- the same signs byte masked with all 8 selector bits.)
      std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
      mlir::Value signsBcast = emitOpaqueCallBuilt(
          rewriter, loc, u8CoreType, bcastCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signs, sizeLit(groupLanes)};
          });
      std::string andCallee = "__riscv_vand_vv_u8m1";
      mlir::Value signBits = emitOpaqueCallBuilt(
          rewriter, loc, u8CoreType, andCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signsBcast, kmask, sizeLit(groupLanes)};
          });
      std::string msneCallee =
          riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
      mlir::Value signMask = emitOpaqueCallBuilt(
          rewriter, loc, maskType, msneCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signBits, intLit(0), sizeLit(groupLanes)};
          });

      // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign.)
      std::string negCallee = "__riscv_vneg_v_i8m1";
      mlir::Value gridNeg = emitOpaqueCallBuilt(
          rewriter, loc, i8CoreType, negCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridV, sizeLit(groupLanes)};
          });
      std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
      mlir::Value gridSigned = emitOpaqueCallBuilt(
          rewriter, loc, i8CoreType, mergeCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridV, gridNeg, signMask, sizeLit(groupLanes)};
          });

      // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
      // <= 62*127 = 7874 < 32767, fits i16.)
      std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
      mlir::Value product = emitOpaqueCallBuilt(
          rewriter, loc, i16WideType, wmulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridSigned, q8V, sizeLit(groupLanes)};
          });

      // sumiAcc = vwredsum(product, sumiAcc, 8);  (chain the i32 reduction;
      // integer add is order-free.)
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, reduceCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {product, sumiAcc, sizeLit(groupLanes)};
          });
    };

    // d = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);  (ONCE
    // per super-block; the fp16 weight scale times the fp32 q8_K scale).
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

    // === PAIR-BATCHED grid gather (the AVL=2 fractional-LMUL fix) ===
    // The 32 per-group vl=2 __riscv_vluxei16_v_i32m1 gathers -- each preceded by a vl=2
    // u16mf2 index load, the fractional-LMUL 2-element scalarization storm -- are HOISTED
    // to ONE wide gather per SUB-BLOCK PAIR. A pair = 2 sub-blocks = 8 sign groups = 16
    // grid u32 indices. The pair index array is laid out 4 u16 slots PER GROUP (2 real
    // idx*4 byte-offsets + 2 zero pads) so that -- after the i32m8 -> i8m8 reinterpret --
    // each group's 2 gathered grid u32 entries (8 bytes) occupy lanes 0..7 of a DISTINCT
    // i8m1 register, recovered by a register-group vget (the pads fill lanes 8..15, which
    // the old per-group vl=2 gather also left unread). ONE vle16_v_u16m4 + ONE
    // vluxei16_v_i32m8 replace 8 vl=2 loads + 8 vl=2 gathers per pair; the per-group
    // sign-fold + widening dot is BYTE-IDENTICAL, and bsum accumulates in STRICT ascending
    // (sub-block, group) order.
    int64_t numPairs = numSubBlocks / 2;       // 4
    int64_t groupsPerPair = 2 * numGroups;     // 8
    int64_t pairIdxSlots = 4 * groupsPerPair;  // 32 (4 slots/group: 2 real + 2 pad)
    int64_t groupElems = subBlock / numGroups; // 32/4 = 8 q8 activations per group
    llvm::StringRef pairGatherLmul = "m8";
    llvm::StringRef pairIdxLmul = "m4";
    mlir::Type i32PairType = emitc::OpaqueType::get(ctx, "vint32m8_t");
    mlir::Type i8PairType = emitc::OpaqueType::get(ctx, "vint8m8_t");
    mlir::Type u16IdxPairType = emitc::OpaqueType::get(ctx, "vuint16m4_t");
    mlir::Type idxArrayPairType =
        emitc::ArrayType::get({pairIdxSlots}, u16ElemType);
    std::string vgetCallee = "__riscv_vget_v_i8m8_i8m1";

    for (int64_t pair = 0; pair < numPairs; ++pair) {
      // --- decode BOTH sub-blocks' aux32/ls and FILL the pair grid-index array ---
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("gridoff", opName, role));
      auto gridOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto gridOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(gridOffVar.getResult());
      auto storeU16 = [&](int64_t slot, mlir::Value u16Val) {
        mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(slot));
        mlir::Value slotElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, gridOffArray,
                                            mlir::ValueRange{slotIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
            u16Val);
      };
      auto storeIdxOff = [&](int64_t slot, mlir::Value idx) {
        mlir::Value byteOff =
            rewriter
                .create<emitc::BitwiseLeftShiftOp>(loc, intType, idx, intLit(2))
                .getResult();
        mlir::Value byteOffU16 =
            rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff).getResult();
        storeU16(slot, byteOffU16);
      };
      mlir::Value u16Zero =
          rewriter.create<emitc::CastOp>(loc, u16ElemType, intLit(0)).getResult();

      mlir::Value lsPair[2];
      mlir::Value aux32Pair[2];
      for (int64_t half = 0; half < 2; ++half) {
        int64_t ib32 = 2 * pair + half;
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
        // uint32_t aux32 = a[0] | a[1]<<8 | a[2]<<16 | a[3]<<24;  (little-endian,
        // uint domain -> logical >>).
        mlir::Value aux32 = loadByteAsUint(aBase, 0);
        aux32 = uOr(aux32, uShl(loadByteAsUint(aBase, 1), uintLit(8)));
        aux32 = uOr(aux32, uShl(loadByteAsUint(aBase, 2), uintLit(16)));
        aux32 = uOr(aux32, uShl(loadByteAsUint(aBase, 3), uintLit(24)));
        aux32Pair[half] = aux32;
        // int ls = 2*(aux32 >> 28) + 1;  (the per-sub-block 4-bit scale, [1,31]).
        mlir::Value lsTop =
            rewriter.create<emitc::CastOp>(loc, intType, uShr(aux32, uintLit(28)))
                .getResult();
        lsPair[half] =
            rewriter
                .create<emitc::AddOp>(
                    loc, intType,
                    rewriter.create<emitc::MulOp>(loc, intType, lsTop, intLit(2))
                        .getResult(),
                    intLit(1))
                .getResult();
        // const uint8_t *qg = q3 + ib32*8;  (8 grid index bytes; 2 per sign group).
        // Fill slots [g*4+{0,1}] with the two idx*4 byte-offsets, [g*4+{2,3}] with zero
        // pads (g = half*numGroups + l), so each group aligns to an i8m1 register.
        mlir::Value qgBase =
            (ib32 == 0)
                ? q3Base
                : rewriter
                      .create<emitc::AddOp>(loc, u8PtrType, q3Base,
                                            sizeLit(ib32 * indicesPerSubBlock))
                      .getResult();
        for (int64_t l = 0; l < numGroups; ++l) {
          int64_t g = half * numGroups + l;
          mlir::Value idx1 = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qgBase, 2 * l + 0);
          mlir::Value idx2 = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qgBase, 2 * l + 1);
          storeIdxOff(g * 4 + 0, idx1);
          storeIdxOff(g * 4 + 1, idx2);
          storeU16(g * 4 + 2, u16Zero);
          storeU16(g * 4 + 3, u16Zero);
        }
      }

      // --- ONE wide vluxei16 gather over the whole pair (16 real + 16 pad slots) ---
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "grid_sign_subblock"));
      mlir::Value idxBaseIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value idxBaseElem0 =
          rewriter
              .create<emitc::SubscriptOp>(loc, gridOffArray,
                                          mlir::ValueRange{idxBaseIndex0})
              .getResult();
      mlir::Value idxBase =
          rewriter
              .create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", idxBaseElem0)
              .getResult();
      std::string idxLoadCallee =
          riscvIntrinsicName("vle", 16, pairIdxLmul, "u16");
      mlir::Value vGridOff = emitOpaqueCallBuilt(
          rewriter, loc, u16IdxPairType, idxLoadCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {idxBase, sizeLit(pairIdxSlots)};
          });
      std::string gatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i32", pairGatherLmul);
      mlir::Value gridGathered = emitOpaqueCallBuilt(
          rewriter, loc, i32PairType, gatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {grid32, vGridOff, sizeLit(pairIdxSlots)};
          });
      std::string reinterpretCallee = "__riscv_vreinterpret_v_i32m8_i8m8";
      mlir::Value gridPairV =
          emitOpaqueCall(rewriter, loc, i8PairType, reinterpretCallee,
                         mlir::ValueRange{gridGathered}, opName, role);

      // --- per-sub-block integer dot (STRICT ascending (sub-block, group) order) ---
      for (int64_t half = 0; half < 2; ++half) {
        int64_t ib32 = 2 * pair + half;
        // int32_t sumi seed (chained i32m1 reduction over the 4 groups; order-free).
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroSeed =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zeroSeed, sizeLit(1)};
            });
        mlir::Value q8Group =
            (ib32 == 0)
                ? q8Base
                : rewriter
                      .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                            sizeLit(ib32 * subBlock))
                      .getResult();
        for (int64_t l = 0; l < numGroups; ++l) {
          int64_t g = half * numGroups + l;
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "grid_sign_group"));
          // int signs = weft_iq3xxs_ksigns[(aux32 >> 7*l) & 127];  (the shift is
          // logical in the uint32_t domain; cast the [0,127] selector to int for the
          // table subscript). REUSES the ksigns sign plane.
          mlir::Value signSel =
              rewriter
                  .create<emitc::CastOp>(
                      loc, intType,
                      uAnd(uShr(aux32Pair[half], uintLit(7 * l)), uintLit(127)))
                  .getResult();
          mlir::Value ksignsName = rewriter.create<emitc::LiteralOp>(
              loc, u8PtrType, "weft_iq3xxs_ksigns");
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

          // gridV = vget(gridPairV, g)  -- this group's 8 signed grid bytes in lanes
          // 0..7 (byte-identical to the old per-group vl=2 gather + reinterpret).
          mlir::Value gridV = emitOpaqueCall(
              rewriter, loc, i8CoreType, vgetCallee,
              mlir::ValueRange{gridPairV, sizeLit(g)}, opName, role);
          sumiAcc = gridOf4Dot(gridV, signs, q8Group, sumiAcc);

          // q8Group += 8 (advance to the next group's 8 activations).
          q8Group =
              rewriter
                  .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                        sizeLit(groupElems))
                  .getResult();
        }

        // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (the sub-block dot).
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value sumi =
            emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                           mlir::ValueRange{sumiAcc}, opName, role);

        // bsum = bsum + sumi * ls;  (integer accumulation; strict order).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, lsPair[half])
                .getResult();
        mlir::Value sumiLs =
            rewriter.create<emitc::MulOp>(loc, i32Type, sumi, lsI32).getResult();
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

// M-FLAT iq3_s super-block GRID-of-4 EXPLICIT-SIGNS byte-exact SHARED body anchor.
// Extracted from the (now-retired) monolith emitIQ3SQ8KBlockDot as a pure code move (the
// emitted C is byte-identical) so the front-door-constructed typed super-block SCALAR-grid
// loop (fold_model "scalar_delta_grid", stride 110) lowers byte-identically by
// construction: same per-super-block body, same facts, same order. It emits ONE
// super-block's body at the current insertion point INSIDE an already-open super-block loop
// whose per-super-block bases xb/yb are provided (the grid/kmask decls, the sumf/nb setup,
// the ONCE 8-lane kmask load + grid32 view, the outer loop, and the trailing `*s = sumf`
// store -- iq3_s applies NO trailing factor -- live in the wrapper). The emitc element/
// pointer types + the load helpers are re-derived here from the MLIRContext (uniqued -> the
// SAME Type instances) so the emit is byte-identical to the monolith's inline body. iq3_s is
// the iq3_xxs GRID-of-4 sibling with the qh 9th-bit inject + explicit per-sub-block signs
// region + explicit two-nibble scales swapped in.
void VariantToEmitCFunc::emitIQ3SSuperBlockGridBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const IQ3SGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    int64_t weightDOffset = cx.weightDOffset;          //   0
    int64_t qsOffset = cx.qsOffset;                    //   2
    int64_t qhOffset = cx.qhOffset;                    //  66
    int64_t signsOffset = cx.signsOffset;              //  74
    int64_t scalesOffset = cx.scalesOffset;            // 106
    int64_t activationDOffset = cx.activationDOffset;  //   0
    int64_t q8Offset = cx.q8Offset;                    //   4
    int64_t subBlock = cx.subBlock;                    //  32
    int64_t numSubBlocks = cx.numSubBlocks;            //   8
    int64_t numGroups = cx.numGroups;                  //   4
    int64_t indicesPerSubBlock = cx.indicesPerSubBlock;//   8
    int64_t signsPerSubBlock = cx.signsPerSubBlock;    //   4
    mlir::Value grid32 = cx.grid32;
    mlir::Value kmask = cx.kmask;

    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");

    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, "vuint8m1_t");
    mlir::Type maskType = emitc::OpaqueType::get(ctx, "vbool8_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    int64_t groupLanes = 8;    // 8 grid lanes per sign group (the two old 4-lane passes)
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used for each grid index byte, qh byte, sign byte, and scale byte;
    // alignment-safe and the values are small/positive so no sign-extension hazard).
    // Emitted by the shared emitLoadByteAsInt(constU8Type, intType) helper.

    // ONE grid-of-4 DOT: the per-group sign-fold + signed widening dot, fed a
    // PRE-GATHERED gridV (this group's 8 signed grid bytes in lanes 0..7, recovered by a
    // register-group vget from the SUPER-BLOCK-PAIR-batched vluxei16 gather emitted in the
    // pair loop below -- the qh injection that assembles the two grid indices is done
    // OUTSIDE this helper, during the index-array fill). The sign-fold ops (vmv/vand/vmsne/
    // vneg/vmerge), the widening product, and the chained vwredsum are BYTE-IDENTICAL to
    // the unbatched body -- only the per-group vl=2 __riscv_vluxei16_v_i32m1 grid gather
    // (the fractional-LMUL 2-element scalarization) is hoisted out. Lane mapping UNCHANGED:
    // lanes 0..3 = grid1/q8[0..3]/kmask{1,2,4,8} (old pass A), lanes 4..7 = grid2/q8[4..7]/
    // kmask{16,32,64,128} (old pass B); per-lane product byte-identical, reduction order-free.
    auto gridOf4Dot = [&](mlir::Value gridV, mlir::Value signs, mlir::Value q8Ptr,
                          mlir::Value sumiAcc) -> mlir::Value {

      // vint8m1_t q8v = __riscv_vle8_v_i8m1(q8Ptr, 8);  (the 8 group activations.)
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      mlir::Value q8V = emitOpaqueCallBuilt(
          rewriter, loc, i8CoreType, i8LoadCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {q8Ptr, sizeLit(groupLanes)};
          });

      // sign-bit mask: m = vmsne(vand(vmv(signs), kmask), 0).  (UNCHANGED ops, vl=8.)
      std::string bcastCallee = "__riscv_vmv_v_x_u8m1";
      mlir::Value signsBcast = emitOpaqueCallBuilt(
          rewriter, loc, u8CoreType, bcastCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signs, sizeLit(groupLanes)};
          });
      std::string andCallee = "__riscv_vand_vv_u8m1";
      mlir::Value signBits = emitOpaqueCallBuilt(
          rewriter, loc, u8CoreType, andCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signsBcast, kmask, sizeLit(groupLanes)};
          });
      std::string msneCallee =
          riscvMaskNonzeroIntrinsicName(8, coreLmul, "u8", 8);
      mlir::Value signMask = emitOpaqueCallBuilt(
          rewriter, loc, maskType, msneCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signBits, intLit(0), sizeLit(groupLanes)};
          });

      // g = vmerge(grid, vneg(grid), m);  (apply the per-lane sign.)
      std::string negCallee = "__riscv_vneg_v_i8m1";
      mlir::Value gridNeg = emitOpaqueCallBuilt(
          rewriter, loc, i8CoreType, negCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridV, sizeLit(groupLanes)};
          });
      std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
      mlir::Value gridSigned = emitOpaqueCallBuilt(
          rewriter, loc, i8CoreType, mergeCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridV, gridNeg, signMask, sizeLit(groupLanes)};
          });

      // p = vwmul_i16m2(gridSigned, q8_v);  (signed widening product, each lane
      // <= 15*127 = 1905 < 32767, fits i16.)
      std::string wmulCallee = ("__riscv_vwmul_vv_i16" + wideLmul).str();
      mlir::Value product = emitOpaqueCallBuilt(
          rewriter, loc, i16WideType, wmulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridSigned, q8V, sizeLit(groupLanes)};
          });

      // sumiAcc = vwredsum(product, sumiAcc, 8);  (chain the i32 reduction;
      // integer add is order-free.)
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      return emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, reduceCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {product, sumiAcc, sizeLit(groupLanes)};
          });
    };

      // d = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);  (ONCE
      // per super-block; the fp16 weight scale times the fp32 q8_K scale).
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

      // === PAIR-BATCHED grid gather (the AVL=2 fractional-LMUL qh fix) ===
      // The 32 per-group vl=2 __riscv_vluxei16_v_i32m1 gathers of qh-injected grid
      // indices -- each preceded by a vl=2 u16mf2 index load, the fractional-LMUL
      // 2-element scalarization storm (iq3_s's heaviest gap) -- are HOISTED to ONE wide
      // gather per SUB-BLOCK PAIR. A pair = 2 sub-blocks = 8 sign groups = 16 qh-injected
      // grid u32 indices. The pair index array is laid out 4 u16 slots PER GROUP (2 real
      // idx*4 byte-offsets + 2 zero pads) so that -- after the i32m8 -> i8m8 reinterpret
      // -- each group's 2 gathered grid u32 entries (8 bytes) occupy lanes 0..7 of a
      // DISTINCT i8m1 register, recovered by a register-group vget (the pads fill the
      // unread lanes 8..15). ONE vle16_v_u16m4 + ONE vluxei16_v_i32m8 replace 8 vl=2
      // loads + 8 vl=2 gathers per pair; the qh injection and the per-group explicit-sign
      // fold + widening dot are BYTE-IDENTICAL, and bsum accumulates in STRICT ascending
      // (sub-block, group) order.
      int64_t numPairs = numSubBlocks / 2;       // 4
      int64_t groupsPerPair = 2 * numGroups;     // 8
      int64_t pairIdxSlots = 4 * groupsPerPair;  // 32 (4 slots/group: 2 real + 2 pad)
      int64_t groupElems = subBlock / numGroups; // 32/4 = 8 q8 activations per group
      llvm::StringRef pairGatherLmul = "m8";
      llvm::StringRef pairIdxLmul = "m4";
      mlir::Type i32PairType = emitc::OpaqueType::get(ctx, "vint32m8_t");
      mlir::Type i8PairType = emitc::OpaqueType::get(ctx, "vint8m8_t");
      mlir::Type u16IdxPairType = emitc::OpaqueType::get(ctx, "vuint16m4_t");
      mlir::Type idxArrayPairType =
          emitc::ArrayType::get({pairIdxSlots}, u16ElemType);
      std::string vgetCallee = "__riscv_vget_v_i8m8_i8m1";

      for (int64_t pair = 0; pair < numPairs; ++pair) {
        // --- decode BOTH sub-blocks' scale + qh and FILL the pair grid-index array ---
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("gridoff", opName, role));
        auto gridOffVar = rewriter.create<emitc::VariableOp>(
            loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
        auto gridOffArray = llvm::cast<mlir::TypedValue<emitc::ArrayType>>(
            gridOffVar.getResult());
        auto storeU16 = [&](int64_t slot, mlir::Value u16Val) {
          mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
              loc, rewriter.getIndexType(), std::to_string(slot));
          mlir::Value slotElem =
              rewriter
                  .create<emitc::SubscriptOp>(loc, gridOffArray,
                                              mlir::ValueRange{slotIdx})
                  .getResult();
          rewriter.create<emitc::AssignOp>(
              loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
              u16Val);
        };
        auto storeIdxOff = [&](int64_t slot, mlir::Value idx) {
          mlir::Value byteOff =
              rewriter
                  .create<emitc::BitwiseLeftShiftOp>(loc, intType, idx,
                                                     intLit(2))
                  .getResult();
          mlir::Value byteOffU16 =
              rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff)
                  .getResult();
          storeU16(slot, byteOffU16);
        };
        mlir::Value u16Zero =
            rewriter.create<emitc::CastOp>(loc, u16ElemType, intLit(0))
                .getResult();

        mlir::Value lsPair[2];
        for (int64_t half = 0; half < 2; ++half) {
          int64_t ib32 = 2 * pair + half;
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "sub_block_explicit_scale"));
          // int sc = sc_base[ib32/2];  ls = ib32 even ? 2*(sc&0xf)+1 : 2*(sc>>4)+1.
          mlir::Value scByte = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, scBase, ib32 / 2);
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
          lsPair[half] =
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
          mlir::Value qhByte = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qhBase, ib32);

          // Fill slots [g*4+{0,1}] with the two qh-injected idx*4 byte-offsets,
          // [g*4+{2,3}] with zero pads (g = half*numGroups + l):
          //   idx1 = qs[ib32*8 + 2l+0] | ((qhb << (8-2l)) & 256);  (pass A)
          //   idx2 = qs[ib32*8 + 2l+1] | ((qhb << (7-2l)) & 256);  (pass B)
          for (int64_t l = 0; l < numGroups; ++l) {
            int64_t g = half * numGroups + l;
            mlir::Value qsByte1 =
                emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qsBase, ib32 * indicesPerSubBlock + 2 * l + 0);
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
                emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qsBase, ib32 * indicesPerSubBlock + 2 * l + 1);
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

            storeIdxOff(g * 4 + 0, idx1);
            storeIdxOff(g * 4 + 1, idx2);
            storeU16(g * 4 + 2, u16Zero);
            storeU16(g * 4 + 3, u16Zero);
          }
        }

        // --- ONE wide vluxei16 gather over the whole pair (16 real + 16 pad slots) ---
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "grid_sign_subblock"));
        mlir::Value idxBaseIndex0 =
            rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
        mlir::Value idxBaseElem0 =
            rewriter
                .create<emitc::SubscriptOp>(loc, gridOffArray,
                                            mlir::ValueRange{idxBaseIndex0})
                .getResult();
        mlir::Value idxBase =
            rewriter
                .create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", idxBaseElem0)
                .getResult();
        std::string idxLoadCallee =
            riscvIntrinsicName("vle", 16, pairIdxLmul, "u16");
        mlir::Value vGridOff = emitOpaqueCallBuilt(
            rewriter, loc, u16IdxPairType, idxLoadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {idxBase, sizeLit(pairIdxSlots)};
            });
        std::string gatherCallee = riscvIndexedMemoryIntrinsicName(
            "vluxei", 16, "i32", pairGatherLmul);
        mlir::Value gridGathered = emitOpaqueCallBuilt(
            rewriter, loc, i32PairType, gatherCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {grid32, vGridOff, sizeLit(pairIdxSlots)};
            });
        std::string reinterpretCallee = "__riscv_vreinterpret_v_i32m8_i8m8";
        mlir::Value gridPairV =
            emitOpaqueCall(rewriter, loc, i8PairType, reinterpretCallee,
                           mlir::ValueRange{gridGathered}, opName, role);

        // --- per-sub-block integer dot (STRICT ascending (sub-block, group) order) ---
        for (int64_t half = 0; half < 2; ++half) {
          int64_t ib32 = 2 * pair + half;
          // int32_t sumi seed (chained i32m1 reduction over the 4 groups; order-free).
          std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          mlir::Value sumiAcc = emitOpaqueCallBuilt(
              rewriter, loc, i32m1Type, seedCallee, opName, role,
              [&](mlir::OpBuilder &b,
                  mlir::Location l) -> llvm::SmallVector<mlir::Value> {
                mlir::Value zeroSeed =
                    rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                        .getResult();
                return {zeroSeed, sizeLit(1)};
              });
          mlir::Value q8Group =
              (ib32 == 0)
                  ? q8Base
                  : rewriter
                        .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                              sizeLit(ib32 * subBlock))
                        .getResult();
          for (int64_t l = 0; l < numGroups; ++l) {
            int64_t g = half * numGroups + l;
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "grid_sign_group"));
            // int signs = sgn[ib32*4 + l];  (the EXPLICIT sign byte read DIRECTLY
            // from the signs region at xb+74 -- NO ksigns lookup).
            mlir::Value signs =
                emitLoadByteAsInt(rewriter, loc, constU8Type, intType, sgnBase, ib32 * signsPerSubBlock + l);

            // gridV = vget(gridPairV, g)  -- this group's 8 signed grid bytes in
            // lanes 0..7 (byte-identical to the old per-group vl=2 gather + reinterpret).
            mlir::Value gridV = emitOpaqueCall(
                rewriter, loc, i8CoreType, vgetCallee,
                mlir::ValueRange{gridPairV, sizeLit(g)}, opName, role);
            sumiAcc = gridOf4Dot(gridV, signs, q8Group, sumiAcc);

            // q8Group += 8 (advance to the next group's 8 activations).
            q8Group =
                rewriter
                    .create<emitc::AddOp>(loc, i8PtrType, q8Group,
                                          sizeLit(groupElems))
                    .getResult();
          }

          // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sumiAcc);  (the sub-block dot).
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          mlir::Value sumi =
              emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                             mlir::ValueRange{sumiAcc}, opName, role);

          // bsum = bsum + sumi * ls;  (integer accumulation; strict order).
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "bsum_accumulate"));
          mlir::Value bsumCur =
              rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
          mlir::Value lsI32 =
              rewriter.create<emitc::CastOp>(loc, i32Type, lsPair[half])
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

// M-FLAT iq2_xs super-block per-half-scale GRID byte-exact SHARED body anchor. Extracted
// from the (now-retired) monolith emitIQ2XSQ8KBlockDot as a pure code move (the emitted C
// is byte-identical) so the front-door-constructed typed super-block SCALAR-grid loop
// (fold_model "scalar_delta_grid", stride 74) lowers byte-identically by construction:
// same per-super-block body, same facts, same order. It emits ONE super-block's body at
// the current insertion point INSIDE an already-open super-block loop whose per-super-block
// bases xb/yb are provided (the grid/signs64 decls, the sumf/nb setup, the ONCE grid64 +
// signs64 i64 views, the outer loop, and the trailing `*s = 0.125f*sumf` store live in the
// wrapper). The emitc element/pointer types + the load helpers are re-derived here from the
// MLIRContext (uniqued -> the SAME Type instances) so the emit is byte-identical to the
// monolith's inline body. UNLIKE iq2_xxs there is NO integer_core_lmul gearbox: the per-half
// body runs at a FIXED 16-lane shape (i64m1 gather + i8m1 view + i16m2 widen + u16mf4 index)
// because the two distinct per-half scales ls1/ls2 force the 16-lane (not 32-lane) collapse.
void VariantToEmitCFunc::emitIQ2XSSuperBlockGridBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const IQ2XSGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    int64_t weightDOffset = cx.weightDOffset;          //   0
    int64_t qsOffset = cx.qsOffset;                    //   2
    int64_t scalesOffset = cx.scalesOffset;            //  66
    int64_t activationDOffset = cx.activationDOffset;  //   0
    int64_t q8Offset = cx.q8Offset;                    //   4
    int64_t subBlock = cx.subBlock;                    //  32
    int64_t numSubBlocks = cx.numSubBlocks;            //   8
    int64_t numGroupsPerHalf = cx.numGroupsPerHalf;    //   2
    mlir::Value gridName = cx.gridName;
    mlir::Value signs64 = cx.signs64;

    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
    // The 9-bit index `w & 511` and the sign selector `w >> 9` must be computed in
    // the UNSIGNED domain (ggml reads q2[l] as a uint16_t) so the >> is a LOGICAL
    // shift -- a signed `int` w with bit 15 set would arithmetic-shift and corrupt
    // the sign selector.
    mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");

    // The per-half DOT runs at a FIXED 16-lane shape (the two distinct per-half scales
    // ls1/ls2 force the 16-lane collapse -- NOT 32-lane like iq2_xxs): the product widens
    // to i16m2 and the reduction is i16m2 -> i32m1. The GATHER, however, is BATCHED per
    // SUB-BLOCK PAIR (2 sub-blocks = 4 halves = 64 lanes): ONE i64m4 grid gather + ONE
    // i64m4 sign gather (8 u64 entries = 8 groups) + ONE i8m4 q8 pair load + ONE i8m4
    // vmul sign-fold, then each 16-lane half is recovered by a register-group vget
    // (i8m4 -> i8m1) and fed to the UNCHANGED per-half vwmul_vv_i16m2 + vwredsum. The pair
    // u16 index EMUL is (16/64)*m4 = m1. (The vget half-recovery assumes the board VLEN=128
    // so one i8m1 register == one 16-lane half; the per-half dot itself stays AVL=16.)
    int64_t halfLanes = 16; // 2 grid entries * 8 i8 = one 16-lane half
    mlir::Type i8WideType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WidestType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    // The batched pair gather/fold register types: the 2*core = m4 wide gather/load/fold
    // and its u16 index EMUL m1; the vget recovers each 16-lane half as an i8m1.
    mlir::Type i8PairType = emitc::OpaqueType::get(ctx, "vint8m4_t");
    mlir::Type i64PairType = emitc::OpaqueType::get(ctx, "vint64m4_t");
    mlir::Type u16IdxPairType = emitc::OpaqueType::get(ctx, "vuint16m1_t");

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

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
    auto uintLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
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
    // Emitted by the shared emitLoadByteAsInt(constU8Type, intType) helper.

    // d = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);  (ONCE
    // per super-block; the fp16 weight scale times the fp32 q8_K scale).
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

    // The per-sub-block dot is BATCHED into PAIRS (pair = 0..numSubBlocks/2-1): each pair
    // decodes TWO sub-blocks (s0 = 2*pair, s1 = 2*pair+1) -- FOUR 16-lane halves -- with ONE
    // wider i64m4 grid gather + ONE wider signs64 gather (8 index slots = 8 groups), ONE
    // wider i8m4 q8 pair load, and ONE wider i8m4 vmul sign-fold. Each 16-lane half is then
    // recovered from the wide register via a register-group vget (i8m4 -> i8m1), so the
    // per-half signed widening product + vwredsum stay BYTE-IDENTICAL to the unbatched body
    // (still i16m2 at AVL=16 -- the ls1/ls2 per-half split forbids a 32-lane collapse) while
    // the gather + vsetvli config are hoisted to the super-block pair (16 gathers -> 8,
    // halved vle16/vle8/vmul + config churn). bsum still accumulates in STRICT ascending
    // (sub-block, half) order, each half with its own explicit ls.
    int64_t pairHalves = 4;            // 2 sub-blocks * 2 halves
    int64_t pairLanes = 2 * subBlock;  // 64 continuous q8 activations per pair
    int64_t numGroupsPair = pairHalves * numGroupsPerHalf; // 8 i64 grid/sign entries
    mlir::Type idxArrayPairType = emitc::ArrayType::get({8}, u16ElemType);
    std::string vgetCallee = "__riscv_vget_v_i8m4_i8m1";

    for (int64_t pair = 0; pair < numSubBlocks / 2; ++pair) {
      // Decode BOTH sub-blocks of the pair up front: the per-half explicit scales (ls1/ls2
      // per sub-block -> lsPair[0..3]) and the 8-slot grid/sign byte-offset index arrays
      // (slots [ph*numGroupsPerHalf + lInHalf]). These are pure integer decode ops --
      // reordering across the two halves changes no value; the gather reads the fully-filled
      // arrays.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "sub_block_explicit_scales"));
      // uint16_t gridoff[8] = { (uint16_t)((w&511)*8), ... };  the 8 grid byte-offsets
      // (idx*8 into the 512-entry u64 grid; max 511*8=4088 < 65535), 4 slots per sub-block.
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("gridoff", opName, role));
      auto gridOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto gridOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(gridOffVar.getResult());
      // uint16_t signoff[8] = { (uint16_t)((w>>9)*8), ... };  the 8 sign byte-offsets
      // (sel*8 into the 128-entry u64 signs64; max 127*8=1016).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("signoff", opName, role));
      auto signOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto signOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(signOffVar.getResult());
      auto storeU16 = [&](mlir::TypedValue<emitc::ArrayType> arr, int64_t slot,
                          mlir::Value byteOff) {
        mlir::Value byteOffU16 =
            rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff).getResult();
        mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(slot));
        mlir::Value slotElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{slotIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
            byteOffU16);
      };
      auto buildScale = [&](mlir::Value nibble) -> mlir::Value {
        // ls = 2*nibble + 1, in the int domain.
        return rewriter
            .create<emitc::AddOp>(
                loc, intType,
                rewriter.create<emitc::MulOp>(loc, intType, nibble, intLit(2))
                    .getResult(),
                intLit(1))
            .getResult();
      };

      mlir::Value lsPair[4];
      for (int64_t s = 0; s < 2; ++s) {
        int64_t ib32 = 2 * pair + s;
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
        mlir::Value scByte = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, scBase, ib32);
        mlir::Value scLow =
            rewriter
                .create<emitc::BitwiseAndOp>(loc, intType, scByte, intLit(15))
                .getResult();
        mlir::Value scHigh =
            rewriter
                .create<emitc::BitwiseRightShiftOp>(loc, intType, scByte,
                                                    intLit(4))
                .getResult();
        lsPair[2 * s + 0] = buildScale(scLow);
        lsPair[2 * s + 1] = buildScale(scHigh);
        // Two halves per sub-block (ph = 2*s + hh): h=0 -> groups l=0,1 scaled by ls1;
        // h=1 -> groups l=2,3 by ls2. Each half's 2 groups decode the uint16 word `w`,
        // pack idx*8 / sel*8 byte-offsets into slots [ph*2 + lInHalf].
        for (int64_t hh = 0; hh < 2; ++hh) {
          int64_t ph = 2 * s + hh;
          for (int64_t lInHalf = 0; lInHalf < numGroupsPerHalf; ++lInHalf) {
            int64_t l = 2 * hh + lInHalf;  // group index 0..3 within the sub-block
            // DELTA(b): uint16_t w = a[2*l] | a[2*l+1]<<8;  (2 LE byte loads from the
            // 2-aligned qs stream -- alignment-safe, unsigned so the >> is logical).
            mlir::Value w = loadByteAsUint(aBase, 2 * l);
            w = uOr(w, uShl(loadByteAsUint(aBase, 2 * l + 1), uintLit(8)));
            // DELTA(a): int idx = w & 511;  grid byte-offset = idx*8.
            mlir::Value idx =
                rewriter
                    .create<emitc::CastOp>(loc, intType, uAnd(w, uintLit(511)))
                    .getResult();
            mlir::Value gridByteOff =
                rewriter.create<emitc::MulOp>(loc, intType, idx, intLit(8))
                    .getResult();
            storeU16(gridOffArray, ph * numGroupsPerHalf + lInHalf, gridByteOff);
            // int sel = w >> 9;  sign byte-offset = sel*8 (the shift is logical in the
            // uint32_t domain). signs64[sel*8..] = the OLD ksigns[sel] expanded to +-1.
            mlir::Value sel =
                rewriter
                    .create<emitc::CastOp>(loc, intType, uShr(w, uintLit(9)))
                    .getResult();
            mlir::Value signByteOff =
                rewriter.create<emitc::MulOp>(loc, intType, sel, intLit(8))
                    .getResult();
            storeU16(signOffArray, ph * numGroupsPerHalf + lInHalf, signByteOff);
          }
        }
      }

      // The BATCHED vluxei16 IQ-gather over the whole PAIR: vle16 the 8 u16 grid/sign
      // indices (u16m1 EMUL = (16/64)*m4), TWO __riscv_vluxei16_v_i64m4 gathers over grid64
      // + signs64 (8 u64 entries = both sub-blocks), each reinterpreted to i8m4 (64 grid /
      // 64 +-1 sign bytes), ONE wider vle8_v_i8m4 q8 pair load (CONTINUOUS across the 4
      // halves) + ONE vmul-onto-grid i8m4 sign fold. The gather/config are hoisted to the
      // pair; the per-half dot below is byte-identical.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "grid_sign_half"));
      auto loadIdxPair =
          [&](mlir::TypedValue<emitc::ArrayType> arr) -> mlir::Value {
        mlir::Value base0 =
            rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
        mlir::Value baseElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{base0})
                .getResult();
        mlir::Value basePtr =
            rewriter.create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", baseElem)
                .getResult();
        std::string idxLoadCallee = riscvIntrinsicName("vle", 16, "m1", "u16");
        return emitOpaqueCallBuilt(
            rewriter, loc, u16IdxPairType, idxLoadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {basePtr, sizeLit(numGroupsPair)};
            });
      };
      mlir::Value vGridOff = loadIdxPair(gridOffArray);
      mlir::Value vSignOff = loadIdxPair(signOffArray);

      // vint64m4_t g64 = __riscv_vluxei16_v_i64m4(grid64, vgridoff, 8);  -- the HARDWARE
      // indexed gather of the 8 u64 grid entries, reinterpreted to i8m4 = the 64 signed
      // grid bytes.
      std::string gridGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", "m4");
      mlir::Value gridGathered = emitOpaqueCallBuilt(
          rewriter, loc, i64PairType, gridGatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridName, vGridOff, sizeLit(numGroupsPair)};
          });
      std::string gridReinterpretCallee = "__riscv_vreinterpret_v_i64m4_i8m4";
      mlir::Value gridPairV = emitOpaqueCall(rewriter, loc, i8PairType,
                                             gridReinterpretCallee,
                                             mlir::ValueRange{gridGathered},
                                             opName, role);

      // vint64m4_t s64 = __riscv_vluxei16_v_i64m4(signs64, vsignoff, 8);  -- the HARDWARE
      // indexed gather of the 8 u64 sign entries (= keven_signs_q2xs), reinterpreted to
      // i8m4 = the 64 +-1 sign bytes (lane->byte mapping identical to the unbatched fold).
      std::string signGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", "m4");
      mlir::Value signGathered = emitOpaqueCallBuilt(
          rewriter, loc, i64PairType, signGatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signs64, vSignOff, sizeLit(numGroupsPair)};
          });
      std::string signReinterpretCallee = "__riscv_vreinterpret_v_i64m4_i8m4";
      mlir::Value signsPairV = emitOpaqueCall(rewriter, loc, i8PairType,
                                              signReinterpretCallee,
                                              mlir::ValueRange{signGathered},
                                              opName, role);

      // const int8_t *q8pair = q8 + pair*64;  vint8m4_t q8v = vle8(q8pair, 64) -- the 64
      // activations of this pair's 4 halves; q8 is CONTINUOUS across the sub-block/half
      // boundaries (ggml never resets q8).
      mlir::Value q8PairBase =
          (pair == 0)
              ? q8Base
              : rewriter
                    .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                          sizeLit(pair * pairLanes))
                    .getResult();
      std::string i8PairLoadCallee = riscvIntrinsicName("vle", 8, "m4", "i8");
      mlir::Value q8PairV = emitOpaqueCallBuilt(
          rewriter, loc, i8PairType, i8PairLoadCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {q8PairBase, sizeLit(pairLanes)};
          });

      // vint8m4_t gs = __riscv_vmul_vv_i8m4(grid, signs, 64);  (apply the per-lane +-1 sign
      // to the GRID, NOT q8 -- byte-identical to the unbatched vmul fold. CRITICAL: the sign
      // MUST fold onto the grid, not q8, because the i8 product wraps: q8 can be -128, and
      // vmul(-128,-1) = 128 wraps to -128 (wrong sign) -- whereas grid in [8,43] so
      // grid*(+-1) in [-43,43] never overflows.)
      std::string signMulCallee = "__riscv_vmul_vv_i8m4";
      mlir::Value gridSignedPair = emitOpaqueCallBuilt(
          rewriter, loc, i8PairType, signMulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridPairV, signsPairV, sizeLit(pairLanes)};
          });

      // Per-half dot: recover each 16-lane half via a register-group vget (i8m4 -> i8m1),
      // then the SAME i8m1 -> i16m2 vwmul + vwredsum + extract as the unbatched body, in
      // STRICT (s0-h0, s0-h1, s1-h0, s1-h1) order with each half's own explicit ls.
      for (int64_t ph = 0; ph < pairHalves; ++ph) {
        mlir::Value gridSigned = emitOpaqueCall(
            rewriter, loc, i8WideType, vgetCallee,
            mlir::ValueRange{gridSignedPair, sizeLit(ph)}, opName, role);
        mlir::Value q8V = emitOpaqueCall(rewriter, loc, i8WideType, vgetCallee,
                                         mlir::ValueRange{q8PairV, sizeLit(ph)},
                                         opName, role);

        // p = __riscv_vwmul_vv_i16m2(gridSigned, q8v, 16);  (signed widening product,
        // each lane <= 43*127 = 5461 < 32767, fits i16).
        std::string wmulCallee = "__riscv_vwmul_vv_i16m2";
        mlir::Value product = emitOpaqueCallBuilt(
            rewriter, loc, i16WidestType, wmulCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {gridSigned, q8V, sizeLit(halfLanes)};
            });

        // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
        //   p, vmv_v_x_i32m1(0,1), 16));  -- ONE reduction per half (the i32 sum is
        // order-free so byte-exact).
        std::string seedCallee =
            riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroSeed =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zeroSeed, sizeLit(1)};
            });
        std::string reduceCallee = "__riscv_vwredsum_vs_i16m2_i32m1";
        sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, reduceCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {product, sumiAcc, sizeLit(halfLanes)};
            });
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value sumi = emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                                          mlir::ValueRange{sumiAcc}, opName,
                                          role);

        // bsum = bsum + sumi * lsPair[ph];  (integer accumulation; strict half order;
        // DELTA(c): the explicit per-half scale).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, lsPair[ph])
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

// M-FLAT iq2_s super-block per-half-scale GRID byte-exact SHARED body anchor. Extracted
// from the (now-retired) monolith emitIQ2SQ8KBlockDot as a pure code move (the emitted C
// is byte-identical) so the front-door-constructed typed super-block SCALAR-grid loop
// (fold_model "scalar_delta_grid", stride 82) lowers byte-identically by construction:
// same per-super-block body, same facts, same order. It emits ONE super-block's body at
// the current insertion point INSIDE an already-open super-block loop whose per-super-block
// bases xb/yb are provided (the grid/signs256 decls, the sumf/nb setup, the ONCE grid64 +
// signs256 i64 views, the outer loop, and the trailing `*s = 0.125f*sumf` store live in the
// wrapper). The emitc element/pointer types + the load helpers are re-derived here from the
// MLIRContext (uniqued -> the SAME Type instances) so the emit is byte-identical to the
// monolith's inline body. Like iq2_xs there is NO integer_core_lmul gearbox: the per-half
// body runs at a FIXED 16-lane shape (i64m1 gather + i8m1 view + i16m2 widen + u16mf4 index)
// because the two distinct per-half scales ls1/ls2 force the 16-lane (not 32-lane) collapse.
void VariantToEmitCFunc::emitIQ2SSuperBlockGridBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const IQ2SGridBodyContext &cx, mlir::Value xb, mlir::Value yb,
    mlir::TypedValue<emitc::LValueType> sumfVar) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    mlir::Type sizeType = cx.sizeType;
    mlir::Type weightPtrType = cx.weightPtrType;
    mlir::Type activationPtrType = cx.activationPtrType;
    int64_t weightDOffset = cx.weightDOffset;          //   0
    int64_t qsOffset = cx.qsOffset;                    //   2
    int64_t signsOffset = cx.signsOffset;              //  34
    int64_t qhOffset = cx.qhOffset;                    //  66
    int64_t scalesOffset = cx.scalesOffset;            //  74
    int64_t activationDOffset = cx.activationDOffset;  //   0
    int64_t q8Offset = cx.q8Offset;                    //   4
    int64_t subBlock = cx.subBlock;                    //  32
    int64_t numSubBlocks = cx.numSubBlocks;            //   8
    int64_t groupsPerSub = cx.groupsPerSub;            //   4
    int64_t numGroupsPerHalf = cx.numGroupsPerHalf;    //   2
    mlir::Value gridName = cx.gridName;
    mlir::Value signs256 = cx.signs256;

    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type intType = emitc::OpaqueType::get(ctx, "int");

    // The per-half DOT runs at a FIXED 16-lane shape (the two distinct per-half scales
    // ls1/ls2 force the 16-lane collapse -- NOT 32-lane like iq2_xxs): the product widens
    // to i16m2 and the reduction is i16m2 -> i32m1. The GATHER, however, is BATCHED per
    // SUB-BLOCK PAIR (2 sub-blocks = 4 halves = 64 lanes): ONE i64m4 grid gather + ONE
    // i64m4 sign gather (8 u64 entries = 8 groups) + ONE i8m4 q8 pair load + ONE i8m4
    // vmul sign-fold, then each 16-lane half is recovered by a register-group vget
    // (i8m4 -> i8m1) and fed to the UNCHANGED per-half vwmul_vv_i16m2 + vwredsum. The pair
    // u16 index EMUL is (16/64)*m4 = m1. (The vget half-recovery assumes the board VLEN=128
    // so one i8m1 register == one 16-lane half; the per-half dot itself stays AVL=16.)
    int64_t halfLanes = 16; // 2 grid entries * 8 i8 = one 16-lane half
    mlir::Type i8WideType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WidestType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
    mlir::Type u16PtrTypeMut =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "uint16_t"));
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    // The batched pair gather/fold register types: the 2*core = m4 wide gather/load/fold
    // and its u16 index EMUL m1; the vget recovers each 16-lane half as an i8m1.
    mlir::Type i8PairType = emitc::OpaqueType::get(ctx, "vint8m4_t");
    mlir::Type i64PairType = emitc::OpaqueType::get(ctx, "vint64m4_t");
    mlir::Type u16IdxPairType = emitc::OpaqueType::get(ctx, "vuint16m1_t");

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // int x = (int)a[i];  -- a structured byte load from a `const uint8_t *` then a
    // cast to int (used to read the single qs index byte, the explicit sign byte,
    // the qh-plane byte, and the explicit scale byte alignment-safely).
    // Emitted by the shared emitLoadByteAsInt(constU8Type, intType) helper.

    // d = (float)*(const _Float16 *)(xb + 0) * *(const float *)(yb + 0);  (ONCE
    // per super-block; the fp16 weight scale times the fp32 q8_K scale).
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

    // The per-sub-block dot is BATCHED into PAIRS (pair = 0..numSubBlocks/2-1): each pair
    // decodes TWO sub-blocks (s0 = 2*pair, s1 = 2*pair+1) -- FOUR 16-lane halves -- with ONE
    // wider i64m4 grid gather + ONE wider signs256 gather (8 index slots = 8 groups), ONE
    // wider i8m4 q8 pair load, and ONE wider i8m4 vmul sign-fold. Each 16-lane half is then
    // recovered from the wide register via a register-group vget (i8m4 -> i8m1), so the
    // per-half signed widening product + vwredsum stay BYTE-IDENTICAL to the unbatched body
    // (still i16m2 at AVL=16 -- the ls1/ls2 per-half split forbids a 32-lane collapse) while
    // the gather + vsetvli config are hoisted to the super-block pair (16 gathers -> 8,
    // halved vle16/vle8/vmul + config churn). bsum still accumulates in STRICT ascending
    // (sub-block, half) order, each half with its own explicit ls.
    int64_t pairHalves = 4;            // 2 sub-blocks * 2 halves
    int64_t pairLanes = 2 * subBlock;  // 64 continuous q8 activations per pair
    int64_t numGroupsPair = pairHalves * numGroupsPerHalf; // 8 i64 grid/sign entries
    mlir::Type idxArrayPairType = emitc::ArrayType::get({8}, u16ElemType);
    std::string vgetCallee = "__riscv_vget_v_i8m4_i8m1";

    for (int64_t pair = 0; pair < numSubBlocks / 2; ++pair) {
      // Decode BOTH sub-blocks of the pair up front: the per-half explicit scales (ls1/ls2
      // per sub-block -> lsPair[0..3]), the per-sub-block qh-plane byte, and the 8-slot
      // grid/sign byte-offset index arrays (slots [ph*numGroupsPerHalf + lInHalf]). These
      // are pure integer decode ops -- reordering across the two halves changes no value;
      // the gather reads the fully-filled arrays.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "sub_block_explicit_scales"));
      // uint16_t gridoff[8] = { (uint16_t)(idx*8), ... };  the 8 grid byte-offsets (idx*8
      // into the 1024-entry u64 grid; max 1023*8=8184 < 65535), 4 slots per sub-block.
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("gridoff", opName, role));
      auto gridOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto gridOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(gridOffVar.getResult());
      // uint16_t signoff[8] = { (uint16_t)(signByte*8), ... };  the 8 sign byte-offsets
      // (signByte*8 into the 256-entry u64 signs256; max 255*8=2040).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("signoff", opName, role));
      auto signOffVar = rewriter.create<emitc::VariableOp>(
          loc, idxArrayPairType, emitc::OpaqueAttr::get(ctx, ""));
      auto signOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(signOffVar.getResult());
      auto storeU16 = [&](mlir::TypedValue<emitc::ArrayType> arr, int64_t slot,
                          mlir::Value byteOff) {
        mlir::Value byteOffU16 =
            rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff).getResult();
        mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(slot));
        mlir::Value slotElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{slotIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
            byteOffU16);
      };
      auto buildScale = [&](mlir::Value nibble) -> mlir::Value {
        // ls = 2*nibble + 1, in the int domain.
        return rewriter
            .create<emitc::AddOp>(
                loc, intType,
                rewriter.create<emitc::MulOp>(loc, intType, nibble, intLit(2))
                    .getResult(),
                intLit(1))
            .getResult();
      };

      mlir::Value lsPair[4];
      for (int64_t s = 0; s < 2; ++s) {
        int64_t ib32 = 2 * pair + s;
        // int sc = sc_base[ib32];  ls1 = 2*(sc & 0xf)+1;  ls2 = 2*(sc>>4)+1.
        mlir::Value scByte = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, scBase, ib32);
        mlir::Value scLow =
            rewriter
                .create<emitc::BitwiseAndOp>(loc, intType, scByte, intLit(15))
                .getResult();
        mlir::Value scHigh =
            rewriter
                .create<emitc::BitwiseRightShiftOp>(loc, intType, scByte,
                                                    intLit(4))
                .getResult();
        lsPair[2 * s + 0] = buildScale(scLow);
        lsPair[2 * s + 1] = buildScale(scHigh);

        // DELTA(b): int qhb = qh[ib32];  (the per-sub-block qh-bit plane byte; its
        // 2-bit fields inject the high bits of each group's 10-bit grid index).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_plane_byte"));
        mlir::Value qhByte = emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qhBase, ib32);

        // Two halves per sub-block (ph = 2*s + hh): h=0 -> groups l=0,1 scaled by ls1;
        // h=1 -> groups l=2,3 by ls2. Each half's 2 groups decode idx (qs|qh-high-bits)
        // + the explicit sign byte, packing idx*8 / signByte*8 into slots [ph*2 + lInHalf].
        for (int64_t hh = 0; hh < 2; ++hh) {
          int64_t ph = 2 * s + hh;
          for (int64_t lInHalf = 0; lInHalf < numGroupsPerHalf; ++lInHalf) {
            int64_t l = 2 * hh + lInHalf;  // group index 0..3 within the sub-block
            // DELTA(a)+(b): idx = qs[ib32*4 + l] | ((qhb << (8-2*l)) & 0x300).
            // The low 8 bits are a single qs index byte; the high 2 bits are the
            // (8-2*l)-shifted qh-plane field masked to bits [8,9]. shift = 8,6,4,2
            // for l = 0,1,2,3. The index is computed in the int domain (`qs[]`/`qh[]`
            // are uint8, shift small/positive, result in [0,1023]). grid off = idx*8.
            mlir::Value qsIdxByte =
                emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qsBase, ib32 * groupsPerSub + l);
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
            mlir::Value gridByteOff =
                rewriter.create<emitc::MulOp>(loc, intType, idx, intLit(8))
                    .getResult();
            storeU16(gridOffArray, ph * numGroupsPerHalf + lInHalf, gridByteOff);
            // DELTA(c): int signByte = sgn[ib32*4 + l];  (the EXPLICIT sign byte read
            // DIRECTLY from the sign region at qs+32 -- NO ksigns lookup). signs256[
            // signByte*8..] = that sign byte expanded to +-1. sign off = signByte*8.
            mlir::Value signByte =
                emitLoadByteAsInt(rewriter, loc, constU8Type, intType, sgnBase, ib32 * groupsPerSub + l);
            mlir::Value signByteOff =
                rewriter.create<emitc::MulOp>(loc, intType, signByte, intLit(8))
                    .getResult();
            storeU16(signOffArray, ph * numGroupsPerHalf + lInHalf, signByteOff);
          }
        }
      }

      // The BATCHED vluxei16 IQ-gather over the whole PAIR: vle16 the 8 u16 grid/sign
      // indices (u16m1 EMUL = (16/64)*m4), TWO __riscv_vluxei16_v_i64m4 gathers over grid64
      // + signs256 (8 u64 entries = both sub-blocks), each reinterpreted to i8m4 (64 grid /
      // 64 +-1 sign bytes), ONE wider vle8_v_i8m4 q8 pair load (CONTINUOUS across the 4
      // halves) + ONE vmul-onto-grid i8m4 sign fold. The gather/config are hoisted to the
      // pair; the per-half dot below is byte-identical.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "grid_sign_half"));
      auto loadIdxPair =
          [&](mlir::TypedValue<emitc::ArrayType> arr) -> mlir::Value {
        mlir::Value base0 =
            rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
        mlir::Value baseElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, arr, mlir::ValueRange{base0})
                .getResult();
        mlir::Value basePtr =
            rewriter.create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", baseElem)
                .getResult();
        std::string idxLoadCallee = riscvIntrinsicName("vle", 16, "m1", "u16");
        return emitOpaqueCallBuilt(
            rewriter, loc, u16IdxPairType, idxLoadCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {basePtr, sizeLit(numGroupsPair)};
            });
      };
      mlir::Value vGridOff = loadIdxPair(gridOffArray);
      mlir::Value vSignOff = loadIdxPair(signOffArray);

      // vint64m4_t g64 = __riscv_vluxei16_v_i64m4(grid64, vgridoff, 8);  -- the HARDWARE
      // indexed gather of the 8 u64 grid entries, reinterpreted to i8m4 = the 64 signed
      // grid bytes.
      std::string gridGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", "m4");
      mlir::Value gridGathered = emitOpaqueCallBuilt(
          rewriter, loc, i64PairType, gridGatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridName, vGridOff, sizeLit(numGroupsPair)};
          });
      std::string gridReinterpretCallee = "__riscv_vreinterpret_v_i64m4_i8m4";
      mlir::Value gridPairV = emitOpaqueCall(rewriter, loc, i8PairType,
                                             gridReinterpretCallee,
                                             mlir::ValueRange{gridGathered},
                                             opName, role);

      // vint64m4_t s64 = __riscv_vluxei16_v_i64m4(signs256, vsignoff, 8);  -- the HARDWARE
      // indexed gather of the 8 u64 sign entries, reinterpreted to i8m4 = the 64 +-1 sign
      // bytes (lane->byte mapping identical to the unbatched fold).
      std::string signGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", "m4");
      mlir::Value signGathered = emitOpaqueCallBuilt(
          rewriter, loc, i64PairType, signGatherCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {signs256, vSignOff, sizeLit(numGroupsPair)};
          });
      std::string signReinterpretCallee = "__riscv_vreinterpret_v_i64m4_i8m4";
      mlir::Value signsPairV = emitOpaqueCall(rewriter, loc, i8PairType,
                                              signReinterpretCallee,
                                              mlir::ValueRange{signGathered},
                                              opName, role);

      // const int8_t *q8pair = q8 + pair*64;  vint8m4_t q8v = vle8(q8pair, 64) -- the 64
      // activations of this pair's 4 halves; q8 is CONTINUOUS across the sub-block/half
      // boundaries (ggml never resets q8).
      mlir::Value q8PairBase =
          (pair == 0)
              ? q8Base
              : rewriter
                    .create<emitc::AddOp>(loc, i8PtrType, q8Base,
                                          sizeLit(pair * pairLanes))
                    .getResult();
      std::string i8PairLoadCallee = riscvIntrinsicName("vle", 8, "m4", "i8");
      mlir::Value q8PairV = emitOpaqueCallBuilt(
          rewriter, loc, i8PairType, i8PairLoadCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {q8PairBase, sizeLit(pairLanes)};
          });

      // vint8m4_t gs = __riscv_vmul_vv_i8m4(grid, signs, 64);  (apply the per-lane +-1 sign
      // to the GRID, NOT q8 -- byte-identical to the unbatched vmul fold. CRITICAL: the sign
      // MUST fold onto the grid, not q8, because the i8 product wraps: q8 can be -128, and
      // vmul(-128,-1) = 128 wraps to -128 (wrong sign) -- whereas grid in [8,43] so
      // grid*(+-1) in [-43,43] never overflows.)
      std::string signMulCallee = "__riscv_vmul_vv_i8m4";
      mlir::Value gridSignedPair = emitOpaqueCallBuilt(
          rewriter, loc, i8PairType, signMulCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {gridPairV, signsPairV, sizeLit(pairLanes)};
          });

      // Per-half dot: recover each 16-lane half via a register-group vget (i8m4 -> i8m1),
      // then the SAME i8m1 -> i16m2 vwmul + vwredsum + extract as the unbatched body, in
      // STRICT (s0-h0, s0-h1, s1-h0, s1-h1) order with each half's own explicit ls.
      for (int64_t ph = 0; ph < pairHalves; ++ph) {
        mlir::Value gridSigned = emitOpaqueCall(
            rewriter, loc, i8WideType, vgetCallee,
            mlir::ValueRange{gridSignedPair, sizeLit(ph)}, opName, role);
        mlir::Value q8V = emitOpaqueCall(rewriter, loc, i8WideType, vgetCallee,
                                         mlir::ValueRange{q8PairV, sizeLit(ph)},
                                         opName, role);

        // p = __riscv_vwmul_vv_i16m2(gridSigned, q8v, 16);  (signed widening product,
        // each lane <= 43*127 = 5461 < 32767, fits i16).
        std::string wmulCallee = "__riscv_vwmul_vv_i16m2";
        mlir::Value product = emitOpaqueCallBuilt(
            rewriter, loc, i16WidestType, wmulCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {gridSigned, q8V, sizeLit(halfLanes)};
            });

        // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
        //   p, vmv_v_x_i32m1(0,1), 16));  -- ONE reduction per half (the i32 sum is
        // order-free so byte-exact).
        std::string seedCallee =
            riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroSeed =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zeroSeed, sizeLit(1)};
            });
        std::string reduceCallee = "__riscv_vwredsum_vs_i16m2_i32m1";
        sumiAcc = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, reduceCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {product, sumiAcc, sizeLit(halfLanes)};
            });
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value sumi = emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                                          mlir::ValueRange{sumiAcc}, opName,
                                          role);

        // bsum = bsum + sumi * lsPair[ph];  (integer accumulation; strict half order;
        // the explicit per-half scale).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bsum_accumulate"));
        mlir::Value bsumCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, bsumVar).getResult();
        mlir::Value lsI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, lsPair[ph])
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

// ===========================================================================
// PR-31 · The OWNED REAL-VECTOR iq3_xxs dequantize_row leaf (the dequant
// true-vector emitter first cell). Reuses the SAME vluxei16 grid gather + sign
// fold (vmv/vand/vmsne/vneg/vmerge) idiom the iq3_xxs block-dot vec_dot body
// (emitIQ3XXSSuperBlockGridBody) renders, then replaces the widening-dot tail
// (vwmul + vwredsum) with an int->float convert (vsext_vf4 + vfcvt_f_x_v) + a
// runtime `db` scale (vfmul_vf) + a unit store (vse32). No reduction, no q8
// activation. Byte-exact to dequantize_row_iq3_xxs by construction (see the
// header note). Emits OWNED __riscv_v intrinsics -- the ISSUE-001 reverse.
// ===========================================================================
mlir::LogicalResult VariantToEmitCFunc::emitDequantizeRowIQ3XXSVectorBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, mlir::Value output, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role) const {
  mlir::MLIRContext *ctx = rewriter.getContext();

  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type intType = emitc::OpaqueType::get(ctx, "int");
  mlir::Type uintType = emitc::OpaqueType::get(ctx, "uint32_t");
  mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
  mlir::Type u8PtrType = emitc::PointerType::get(constU8Type);
  mlir::Type inputPtrType = input.getType();   // const uint8_t *
  mlir::Type outputPtrType = output.getType(); // float *
  mlir::Type floatPtrType = emitc::PointerType::get(floatType);
  mlir::Type u16ElemType = emitc::OpaqueType::get(ctx, "uint16_t");
  mlir::Type u16PtrTypeMut = emitc::PointerType::get(u16ElemType);
  mlir::Type u32PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint32_t"));
  mlir::Type i32PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int32_t"));

  // The FIXED iq3_xxs grid-of-4 x 2 geometry: a sign group = 8 grid bytes (2 grid u32
  // entries), a sub-block = 4 sign groups = 32 grid bytes, and `db` is CONSTANT across
  // a sub-block. The naive body renders this as 4 per-group vl=2 __riscv_vluxei16 grid
  // gathers (the fractional-LMUL 2-element scalarization storm) + 4 per-group vl=8 (of
  // 32) float pipelines. This body HOISTS that to ONE wide vl=8 grid gather + ONE
  // full-LMUL vl=32 sign-fold + int->float convert per sub-block -- the SAME HOIST
  // emitIQ3XXSSuperBlockGridBody applies to the vec_dot body (RVVToEmitCGridCodebook
  // pair-batch). All LMULs (u8m2/i8m2/i32m8/f32m8, u16m1/i32m2 for the gather) are
  // DERIVED from the 32-lane sub-block width, NOT tunable knobs.
  const int64_t groupLanes = 8;
  const int64_t numGroups = 4;
  const int64_t subBlockLanes = groupLanes * numGroups; // 32
  mlir::Type u8ElemType = emitc::OpaqueType::get(ctx, "uint8_t");
  mlir::Type u8PtrTypeMut = emitc::PointerType::get(u8ElemType);
  mlir::Type u8WideType = emitc::OpaqueType::get(ctx, "vuint8m2_t");
  mlir::Type i8WideType = emitc::OpaqueType::get(ctx, "vint8m2_t");
  mlir::Type maskWideType = emitc::OpaqueType::get(ctx, "vbool4_t");
  mlir::Type i32GatherType = emitc::OpaqueType::get(ctx, "vint32m2_t");
  mlir::Type u16IdxType = emitc::OpaqueType::get(ctx, "vuint16m1_t");
  mlir::Type i32WideType = emitc::OpaqueType::get(ctx, "vint32m8_t");
  mlir::Type f32WideType = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
  mlir::Type u16ArrayType = emitc::ArrayType::get({groupLanes}, u16ElemType);
  mlir::Type u8Array4Type = emitc::ArrayType::get({numGroups}, u8ElemType);

  llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };
  auto floatLit = [&](llvm::StringRef s) -> mlir::Value {
    return rewriter.create<emitc::LiteralOp>(loc, floatType, s);
  };
  auto uintLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };
  // The aux32 / scale / sign-selector bit ops run in the UNSIGNED domain so the >>
  // is a LOGICAL shift (ggml's aux32 is uint32_t -- a signed >> with bit 31 set
  // would corrupt the scale/selector, the iq2_xxs hardware-bisected bug).
  auto uAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return rewriter.create<emitc::BitwiseAndOp>(loc, uintType, a, b).getResult();
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
  // uint32_t x = (uint32_t)a[i];  -- alignment-safe byte load (NO *(uint32_t*)).
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

  // The route-source provenance token (the emit is DRIVEN by the typed region
  // op-identity threaded through opName/role, NOT the abstract format string).
  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // ---- The function-local static grid / ksigns / kmask decls (ONCE) ----------
  // The 256-entry GRID-of-4 codebook + the 128-entry ksigns sign plane, from the
  // canonical kIQ3XXSGrid / kIQ3XXSKsigns (the SAME anchors the block-dot vec_dot
  // lowering renders). The kmask {1<<j} sign-bit selector is an inline const.
  emitIQ3XXSCanonicalGridTableDecl(rewriter, loc);
  emitIQ3XXSCanonicalKsignsTableDecl(rewriter, loc);
  // The wide sign-fold statics (emitted ONCE, DERIVED from the fixed 8-lane group /
  // 4-group sub-block geometry -- NOT literal knobs): kmask32 = the 8-bit {1<<j}
  // selector pattern REPLICATED across the sub-block's 4 sign groups (lane l*8+j tests
  // bit j of group l's sign byte); sigspread = the sub-block broadcast index
  // (sigspread[l*8+j]=l) that fans the 4 per-group sign bytes to their 8 lanes.
  {
    std::string kmaskDecl = "static const uint8_t weft_iq3xxs_kmask32[" +
                            std::to_string(subBlockLanes) + "] = {";
    std::string spreadDecl = "static const uint8_t weft_iq3xxs_sigspread[" +
                             std::to_string(subBlockLanes) + "] = {";
    for (int64_t g = 0; g < numGroups; ++g) {
      for (int64_t j = 0; j < groupLanes; ++j) {
        if (g || j) {
          kmaskDecl += ", ";
          spreadDecl += ", ";
        }
        kmaskDecl += std::to_string(int64_t(1) << j);
        spreadDecl += std::to_string(g);
      }
    }
    kmaskDecl += "};";
    spreadDecl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, kmaskDecl);
    rewriter.create<emitc::VerbatimOp>(loc, spreadDecl);
  }

  // size_t nb = k / 256;
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "super_block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(256));

  // vuint8m2_t kmask32 = vle8(weft_iq3xxs_kmask32, 32);  (ONCE) -- the full 8-bit
  // selector replicated across the sub-block's 4 groups (the vl=32 sign fold masks
  // each group's signs byte with the {1<<j} pattern in its 8 lanes).
  std::string u8LoadCallee = riscvIntrinsicName("vle", 8, "m2", "u8");
  mlir::Value kmask32 = emitOpaqueCallBuilt(
      rewriter, loc, u8WideType, u8LoadCallee, opName, role,
      [&](mlir::OpBuilder &b,
          mlir::Location l) -> llvm::SmallVector<mlir::Value> {
        mlir::Value kmaskName = rewriter.create<emitc::LiteralOp>(
            loc, u8PtrType, "weft_iq3xxs_kmask32");
        return {kmaskName, sizeLit(subBlockLanes)};
      },
      llvm::StringRef("kmask_table_load"));
  // vuint8m2_t sigspread = vle8(weft_iq3xxs_sigspread, 32);  (ONCE) -- the sub-block
  // broadcast index that fans the 4 per-group sign bytes to their 8 lanes each.
  mlir::Value sigspread = emitOpaqueCallBuilt(
      rewriter, loc, u8WideType, u8LoadCallee, opName, role,
      [&](mlir::OpBuilder &b,
          mlir::Location l) -> llvm::SmallVector<mlir::Value> {
        mlir::Value spreadName = rewriter.create<emitc::LiteralOp>(
            loc, u8PtrType, "weft_iq3xxs_sigspread");
        return {spreadName, sizeLit(subBlockLanes)};
      },
      llvm::StringRef("sigspread_table_load"));

  // const int32_t *grid32 = (const int32_t *)weft_iq3xxs_grid;  (signed-i32 view of
  // the uint32[256] grid for the vluxei16 indexed gather.)
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "grid_table_i32_view"));
  mlir::Value gridArrayName =
      rewriter.create<emitc::LiteralOp>(loc, u32PtrType, "weft_iq3xxs_grid");
  mlir::Value grid32 =
      rewriter.create<emitc::CastOp>(loc, i32PtrType, gridArrayName).getResult();

  // ---- for (size_t ib = 0; ib < nb; ib += 1) -------------------------------
  auto blockFor = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1),
                                                /*bodyBuilder=*/nullptr);
  mlir::Value ib = blockFor.getInductionVar();
  {
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(blockFor.getBody());

    // const uint8_t *xb = x + ib*98;   float *yb = y + ib*256;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "xb"));
    mlir::Value xb =
        rewriter
            .create<emitc::AddOp>(
                loc, inputPtrType, input,
                rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(98)))
            .getResult();
    mlir::Value ybRaw =
        rewriter
            .create<emitc::AddOp>(
                loc, outputPtrType, output,
                rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(256)))
            .getResult();
    mlir::Value yb =
        rewriter.create<emitc::CastOp>(loc, floatPtrType, ybRaw).getResult();

    // float d = (float)*(const _Float16 *)(xb + 0);
    mlir::Value d = emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                                   mlir::ValueRange{xb}, opName, role,
                                   llvm::StringRef("fcvt.s.h"));

    // const uint8_t *q3 = xb + 2;  (64 grid index bytes.)
    // const uint8_t *gas = xb + 66; (32 aux bytes = 8 uint32.)
    mlir::Value q3Base =
        rewriter
            .create<emitc::CastOp>(
                loc, u8PtrType,
                rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(2)))
            .getResult();
    mlir::Value gasBase =
        rewriter
            .create<emitc::CastOp>(
                loc, u8PtrType,
                rewriter.create<emitc::AddOp>(loc, inputPtrType, xb, sizeLit(66)))
            .getResult();

    for (int64_t ib32 = 0; ib32 < 8; ++ib32) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "sub_block_aux_scale"));
      // const uint8_t *a = gas + ib32*4;
      mlir::Value aBase =
          (ib32 == 0)
              ? gasBase
              : rewriter
                    .create<emitc::AddOp>(loc, u8PtrType, gasBase,
                                          sizeLit(ib32 * 4))
                    .getResult();
      // uint32_t aux = a[0] | a[1]<<8 | a[2]<<16 | a[3]<<24;  (LE, uint domain.)
      mlir::Value aux = loadByteAsUint(aBase, 0);
      aux = uOr(aux, uShl(loadByteAsUint(aBase, 1), uintLit(8)));
      aux = uOr(aux, uShl(loadByteAsUint(aBase, 2), uintLit(16)));
      aux = uOr(aux, uShl(loadByteAsUint(aBase, 3), uintLit(24)));
      // float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
      mlir::Value auxTop =
          rewriter.create<emitc::CastOp>(loc, intType, uShr(aux, uintLit(28)))
              .getResult();
      mlir::Value db = rewriter.create<emitc::MulOp>(
          loc, floatType,
          rewriter.create<emitc::MulOp>(
              loc, floatType, d,
              rewriter.create<emitc::AddOp>(
                  loc, floatType, floatLit("0.5f"),
                  rewriter.create<emitc::CastOp>(loc, floatType, auxTop)
                      .getResult()))
              .getResult(),
          floatLit("0.5f"))
                           .getResult();

      // const uint8_t *qg = q3 + ib32*8;  (8 grid index bytes; 2 per sign group.)
      mlir::Value qgBase =
          (ib32 == 0)
              ? q3Base
              : rewriter
                    .create<emitc::AddOp>(loc, u8PtrType, q3Base,
                                          sizeLit(ib32 * 8))
                    .getResult();

      // ===== WIDE sub-block: ONE vl=8 grid gather + ONE vl=32 sign-fold/convert =====
      // The 4 per-group vl=2 gathers are HOISTED to ONE wide vl=8 vluxei16 over all 8
      // grid u32 entries; the 4 per-group vl=8 float pipelines are fused into ONE
      // vl=32 (full i8m2/i32m8/f32m8 LMUL) sign-fold + int->float convert + db scale +
      // unit store. `db` is CONSTANT across the sub-block. Lane l*8+j is byte-identical
      // to the naive body's group l lane j; the store is contiguous, so the output row
      // layout is unchanged.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "grid_sign_subblock"));

      // uint16_t goff[8]; goff[s] = (uint16_t)(qg[s] << 2);  (the 8 grid u32 byte-
      // offsets, contiguous over the sub-block's 4 groups, for the ONE wide gather.)
      auto gridOffVar = rewriter.create<emitc::VariableOp>(
          loc, u16ArrayType, emitc::OpaqueAttr::get(ctx, ""));
      auto gridOffArray =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(gridOffVar.getResult());
      for (int64_t s = 0; s < groupLanes; ++s) {
        mlir::Value idx =
            emitLoadByteAsInt(rewriter, loc, constU8Type, intType, qgBase, s);
        mlir::Value byteOff =
            rewriter
                .create<emitc::BitwiseLeftShiftOp>(loc, intType, idx, intLit(2))
                .getResult();
        mlir::Value byteOffU16 =
            rewriter.create<emitc::CastOp>(loc, u16ElemType, byteOff).getResult();
        mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(s));
        mlir::Value slotElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, gridOffArray,
                                            mlir::ValueRange{slotIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
            byteOffU16);
      }

      // vuint16m1_t voff = vle16(&goff[0], 8);
      mlir::Value idxBaseIndex0 = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), "0");
      mlir::Value idxBaseElem0 =
          rewriter
              .create<emitc::SubscriptOp>(loc, gridOffArray,
                                          mlir::ValueRange{idxBaseIndex0})
              .getResult();
      mlir::Value idxBase =
          rewriter.create<emitc::ApplyOp>(loc, u16PtrTypeMut, "&", idxBaseElem0)
              .getResult();
      std::string idxLoadCallee = riscvIntrinsicName("vle", 16, "m1", "u16");
      mlir::Value voff = emitOpaqueCall(
          rewriter, loc, u16IdxType, idxLoadCallee,
          mlir::ValueRange{idxBase, sizeLit(groupLanes)}, opName, role);

      // vint32m2_t gg = vluxei16_v_i32m2(grid32, voff, 8);  (ONE wide HARDWARE gather
      // over the whole sub-block -- replaces 4 fractional-LMUL vl=2 gathers.)
      std::string gatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 16, "i32", "m2");
      mlir::Value gg = emitOpaqueCall(
          rewriter, loc, i32GatherType, gatherCallee,
          mlir::ValueRange{grid32, voff, sizeLit(groupLanes)}, opName, role);
      // vint8m2_t gridV = vreinterpret_v_i32m2_i8m2(gg);  (32 grid bytes; group l at
      // lanes l*8..l*8+7 == the naive body's per-group lanes 0..7.)
      mlir::Value gridV = emitOpaqueCall(
          rewriter, loc, i8WideType, "__riscv_vreinterpret_v_i32m2_i8m2",
          mlir::ValueRange{gg}, opName, role);

      // uint8_t signs4[4] = { weft_iq3xxs_ksigns[(aux >> 7*l) & 127] } for l=0..3;
      // (the 4 per-group sign selector bytes -- SAME ksigns lookup as the naive body.)
      auto signs4Var = rewriter.create<emitc::VariableOp>(
          loc, u8Array4Type, emitc::OpaqueAttr::get(ctx, ""));
      auto signs4Array =
          llvm::cast<mlir::TypedValue<emitc::ArrayType>>(signs4Var.getResult());
      for (int64_t l = 0; l < numGroups; ++l) {
        mlir::Value signSel =
            rewriter
                .create<emitc::CastOp>(
                    loc, intType, uAnd(uShr(aux, uintLit(7 * l)), uintLit(127)))
                .getResult();
        mlir::Value ksignsName = rewriter.create<emitc::LiteralOp>(
            loc, u8PtrType, "weft_iq3xxs_ksigns");
        mlir::Value signsElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(ksignsName),
                    signSel)
                .getResult();
        mlir::Value signsU8 =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, signsElem)
                .getResult();
        mlir::Value signsMut =
            rewriter.create<emitc::CastOp>(loc, u8ElemType, signsU8).getResult();
        mlir::Value slotIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(l));
        mlir::Value slotElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, signs4Array,
                                            mlir::ValueRange{slotIdx})
                .getResult();
        rewriter.create<emitc::AssignOp>(
            loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(slotElem),
            signsMut);
      }

      // vuint8m2_t signsVec = vluxei8_v_u8m2(signs4, sigspread, 32);  (each group's
      // sign byte broadcast to its 8 lanes -- sigspread[l*8+j]=l, ONE wide gather from
      // the 4-byte sub-block sign array; empirically CHEAPER than a register-only
      // vmv/vmseq/vmerge fan-out on this hardware.)
      mlir::Value s4Elem0 =
          rewriter
              .create<emitc::SubscriptOp>(loc, signs4Array,
                                          mlir::ValueRange{idxBaseIndex0})
              .getResult();
      mlir::Value s4Ptr =
          rewriter.create<emitc::ApplyOp>(loc, u8PtrTypeMut, "&", s4Elem0)
              .getResult();
      mlir::Value s4PtrConst =
          rewriter.create<emitc::CastOp>(loc, u8PtrType, s4Ptr).getResult();
      std::string signsGatherCallee =
          riscvIndexedMemoryIntrinsicName("vluxei", 8, "u8", "m2");
      mlir::Value signsVec = emitOpaqueCall(
          rewriter, loc, u8WideType, signsGatherCallee,
          mlir::ValueRange{s4PtrConst, sigspread, sizeLit(subBlockLanes)}, opName,
          role);

      // sign-bit mask: m = vmsne(vand(signsVec, kmask32), 0), vl=32.  (lane l*8+j =
      // signs[l] & {1<<j} != 0 -- byte-identical to the naive per-group vl=8 fold.)
      mlir::Value signBits = emitOpaqueCall(
          rewriter, loc, u8WideType, "__riscv_vand_vv_u8m2",
          mlir::ValueRange{signsVec, kmask32, sizeLit(subBlockLanes)}, opName,
          role);
      std::string msneCallee = riscvMaskNonzeroIntrinsicName(8, "m2", "u8", 4);
      mlir::Value signMask = emitOpaqueCall(
          rewriter, loc, maskWideType, msneCallee,
          mlir::ValueRange{signBits, intLit(0), sizeLit(subBlockLanes)}, opName,
          role);
      // g = vmerge(grid, vneg(grid), m);  (apply the per-lane +-1 sign, vl=32.)
      mlir::Value gridNeg = emitOpaqueCall(
          rewriter, loc, i8WideType, "__riscv_vneg_v_i8m2",
          mlir::ValueRange{gridV, sizeLit(subBlockLanes)}, opName, role);
      mlir::Value gridSigned = emitOpaqueCall(
          rewriter, loc, i8WideType, "__riscv_vmerge_vvm_i8m2",
          mlir::ValueRange{gridV, gridNeg, signMask, sizeLit(subBlockLanes)},
          opName, role);

      // int->float: gs32 = vsext_vf4(gridSigned); gf = vfcvt_f_x_v(gs32).  (vl=32;
      // all grid bytes < 128, so the signed i8 view == ggml's (uint8_t) read.)
      mlir::Value gs32 = emitOpaqueCall(
          rewriter, loc, i32WideType, "__riscv_vsext_vf4_i32m8",
          mlir::ValueRange{gridSigned, sizeLit(subBlockLanes)}, opName, role);
      std::string cvtCallee = riscvIntrinsicName("vfcvt_f_x_v", 32, "m8", "f32");
      mlir::Value gf = emitOpaqueCall(
          rewriter, loc, f32WideType, cvtCallee,
          mlir::ValueRange{gs32, sizeLit(subBlockLanes)}, opName, role);
      // r = vfmul_vf(gf, db, 32);  (the runtime db scale, constant across the sub-
      // block; db*(+-grid) is byte-exact to (db*grid)*(+-1) -- a float sign flip.)
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, "m8", "f32");
      mlir::Value r = emitOpaqueCall(
          rewriter, loc, f32WideType, mulCallee,
          mlir::ValueRange{gf, db, sizeLit(subBlockLanes)}, opName, role);

      // vse32_v_f32m8(yb + ib32*32, r, 32);  (ONE contiguous 32-float store.)
      int64_t outBase = ib32 * subBlockLanes;
      mlir::Value yptr =
          (outBase == 0)
              ? yb
              : rewriter
                    .create<emitc::AddOp>(loc, floatPtrType, yb, sizeLit(outBase))
                    .getResult();
      std::string vseCallee = riscvIntrinsicName("vse", 32, "m8", "f32");
      emitOpaqueCallVoid(rewriter, loc, vseCallee,
                         mlir::ValueRange{yptr, r, sizeLit(subBlockLanes)}, opName,
                         role);
    }
  }

  return mlir::success();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
