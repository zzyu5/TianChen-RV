#include "RVVToEmitCInternal.h"
#include "Weft/Conversion/RVV/RVVToEmitCSupport.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"

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

// Flat block-dot typed primitive artifact consumers.

mlir::Value VariantToEmitCFunc::emitQ1_0TypedFlatBlockDotBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase,
    mlir::TypedValue<emitc::PointerType> outPointer, mlir::Value avlArg,
    mlir::Type sizeType, llvm::StringRef opName, llvm::StringRef role,
    llvm::StringRef coreLmul, int64_t qk, int64_t weightStride,
    int64_t activationStride, int64_t q8PerWeight, int64_t weightQuantOffset,
    int64_t activationQuantOffset) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The binary sign decode runs ONE 32-lane sub-block body at the whole-LMUL
    // anchor whose i8 VLMAX spans the 32-element sub-block (m2 at VLEN128 where
    // e8m1 VLMAX is 16 < 32; m1 at VLEN256 where e8m1 VLMAX is 32). The gearbox
    // stamps integer_core_lmul from getRVVStripVLMAXElements; the default is m2 --
    // the VLEN-universal-safe floor (e8m2 VLMAX 32 spans the 32-element sub-block
    // at VLEN128 AND VLEN256), so an attr-less op lowers correctly at any VLEN; the
    // gearbox refines m2->m1 only at VLEN>=256. The 4 packed bit-bytes load
    // DIRECTLY into the i8 sign mask (vlm_v_b{ratio}, the packed bits ARE the mask:
    // bit 8b+i -> lane 8b+i), the i8 q8 quants are negated/merged in the i8 domain,
    // and ONE vwredsum widens i8->i16m1 per sub-block (no separate vwcvt; this is
    // ggml's shipped _vl128 lane structure).
    // coreLmul is passed from the constructed q1_0 binary-sign core brick's
    // integer_core_lmul attr (default "m2", the VLEN-universal floor).
    // The vbool ratio is SEW8/LMUL: m1 -> vbool8_t (vlm_v_b8), m2 -> vbool4_t
    // (vlm_v_b4). Derived from the anchor so the mask width tracks the LMUL flip.
    llvm::StringRef boolRatio = (coreLmul == "m2") ? "4" : "8";
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type maskType =
        emitc::OpaqueType::get(ctx, ("vbool" + boolRatio + "_t").str());

    // The block-format structural facts (I4) are passed in: qk 128, weightStride
    // 18, activationStride 34, q8PerWeight 4, weightQuantOffset 2,
    // activationQuantOffset 2. The constructed flat path reads them from the loop
    // op plus the q1_0 binary-sign core brick.
    int64_t subBlockElems = qk / q8PerWeight;            // 32 (q8 block lanes)
    int64_t bytesPerSubBlock = subBlockElems / 8;        // 4 bit bytes per q8 block

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK1_0;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // The ONE sanctioned opaque scalar fp16->fp32 read (a typed emitc.call_opaque
    // node, exactly how the q4_0 sibling emits its fp16 scale reads).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{blockBase}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1) { ... }.
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // const uint8_t *xb = vx + ib*18;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "block_base_x"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(weightStride));
      mlir::Value xb = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                     weightBase, xOff);
      // float d0 = (float)*(const _Float16 *)(xb);
      mlir::Value d0 = fp16Read(xb);

      // float sumi = 0.0f;  (RESET each super-block; the q8-sub-block fp32 fold)
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(floatType),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar,
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

      // The FOUR q8 sub-blocks are UNROLLED (each carries its own d1_k + 32-bit
      // sign decode), folded in strict ascending k order (fp non-associativity).
      for (int64_t k = 0; k < q8PerWeight; ++k) {
        // const uint8_t *yb = vy + (ib*4 + k)*34;
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "block_base_y"));
        mlir::Value ibTimes =
            rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(q8PerWeight));
        mlir::Value yIdx = rewriter.create<emitc::AddOp>(loc, sizeType, ibTimes,
                                                         sizeLit(k));
        mlir::Value yOff = rewriter.create<emitc::MulOp>(loc, sizeType, yIdx,
                                                         sizeLit(activationStride));
        mlir::Value yb = rewriter.create<emitc::AddOp>(loc, activationPtrType,
                                                       activationBase, yOff);
        // float d1 = (float)*(const _Float16 *)(yb);
        mlir::Value d1 = fp16Read(yb);

        // The q8 sub-block is ONE 32-lane body (ggml's shipped _vl128 lane
        // structure): vlm_v_b{ratio} the 4 packed bit-bytes straight into the i8
        // sign mask, vle8 the 32 q8 quants, i8-domain vneg/vmerge -> signed q8,
        // ONE vwredsum widening i8 -> i16m1. No 8-lane sub-grouping, no kmask
        // table, no separate vwcvt: the integer dot is a single 32-lane reduce.

        // size_t vl = __riscv_vsetvl_e8{coreLmul}(32);  (the whole 32-element
        // sub-block; the anchor's i8 VLMAX spans it -- m2 at VLEN128, m1 at
        // VLEN256 -- so vl stays 32 and never crosses into a second sub-block).
        std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, coreLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, setvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(subBlockElems)};
            });

        // const uint8_t *qsbits = xb + 2 + k*4;  (the 4 packed bit-bytes = 32
        // sign bits of this sub-block).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "bits_byte_addr"));
        int64_t bitByteOffset = weightQuantOffset + k * bytesPerSubBlock;
        mlir::Value bitsPtr = rewriter
                                  .create<emitc::AddOp>(loc, weightPtrType, xb,
                                                        sizeLit(bitByteOffset))
                                  .getResult();
        mlir::Value bitsPtrU8 =
            rewriter.create<emitc::CastOp>(loc, u8PtrType, bitsPtr).getResult();

        // vbool{ratio}_t is_not_zero = vlm_v_b{ratio}(qsbits, 32);  (the packed
        // bits ARE the i8 sign mask: bit 8b+i -> lane 8b+i, set -> +q8).
        std::string lmCallee = ("__riscv_vlm_v_b" + boolRatio).str();
        mlir::Value signMask =
            emitOpaqueCall(rewriter, loc, maskType, lmCallee,
                           mlir::ValueRange{bitsPtrU8, vl}, opName, role);

        // vint8{coreLmul}_t q8 = vle8(yb + 2);  (the 32 q8 quants of this block).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "q8_block_addr"));
        mlir::Value q8Ptr = rewriter
                                .create<emitc::AddOp>(loc, activationPtrType, yb,
                                                      sizeLit(activationQuantOffset))
                                .getResult();
        mlir::Value q8PtrI8 =
            rewriter.create<emitc::CastOp>(loc, i8PtrType, q8Ptr).getResult();
        std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        mlir::Value q8 =
            emitOpaqueCall(rewriter, loc, i8CoreType, loadCallee,
                           mlir::ValueRange{q8PtrI8, vl}, opName, role);

        // sy = vmerge(vneg_i8(q8), q8, is_not_zero);  (i8-domain negate/merge,
        // ggml's exact ops: +q8 where bit set, -q8 where clear). The q8 quant
        // domain is [-127,127] (the -128 boundary never occurs in a real q8_0
        // quantization), so the i8 vneg is exact on every gate input.
        std::string negCallee = ("__riscv_vneg_v_i8" + coreLmul).str();
        mlir::Value q8Neg =
            emitOpaqueCall(rewriter, loc, i8CoreType, negCallee,
                           mlir::ValueRange{q8, vl}, opName, role);
        std::string mergeCallee = riscvIntrinsicName("vmerge", 8, coreLmul, "i8");
        mlir::Value signedQ8 =
            emitOpaqueCall(rewriter, loc, i8CoreType, mergeCallee,
                           mlir::ValueRange{q8Neg, q8, signMask, vl}, opName,
                           role);

        // int sumi_block = vmv_x_s(vwredsum_i8{coreLmul}_i16m1(sy, 0, 32));  (ONE
        // widening reduce over the 32 lanes: i8 product chain summed into i16m1.
        // 32 lanes * |q8|<=127 = 4064 < 32767, so the i16 accumulator never
        // overflows.)
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 16, "m1", "i16");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i16m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zeroI16 = rewriter.create<emitc::LiteralOp>(
                  loc, emitc::OpaqueType::get(ctx, "int16_t"), "0");
              return {zeroI16, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i8" + coreLmul + "_i16m1").str();
        mlir::Value red =
            emitOpaqueCall(rewriter, loc, i16m1Type, reduceCallee,
                           mlir::ValueRange{signedQ8, seed, vl}, opName, role);
        std::string extractCallee = "__riscv_vmv_x_s_i16m1_i16";
        mlir::Type i16Type = emitc::OpaqueType::get(ctx, "int16_t");
        mlir::Value sumiBlockI16 =
            emitOpaqueCall(rewriter, loc, i16Type, extractCallee,
                           mlir::ValueRange{red}, opName, role);
        // int32_t sumi_block = (int)..;  (the sub-block integer dot result)
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi_block", opName, role));
        auto sumiBlockVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        mlir::Value sumiBlockI32 =
            rewriter.create<emitc::CastOp>(loc, i32Type, sumiBlockI16)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiBlockVar, sumiBlockI32);

        // sumi = sumi + d1 * (float)sumi_block;  (ggml EXACT order, grouped into
        // ONE emitc.expression so mlir-translate renders ONE C statement and the
        // compiler fuses the same FMA ggml does under -ffp-contract=on/default).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "fp32_accumulate_sub"));
        mlir::Value sumiBlockFinal =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiBlockVar)
                .getResult();
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, floatType, sumiVar).getResult();
        auto subExpr = rewriter.create<emitc::ExpressionOp>(
            loc, floatType, /*do_not_inline=*/false);
        {
          mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
          mlir::Block *exprBlock = rewriter.createBlock(&subExpr.getRegion());
          rewriter.setInsertionPointToStart(exprBlock);
          mlir::Value sumiBlockFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiBlockFinal)
                  .getResult();
          mlir::Value term =
              rewriter.create<emitc::MulOp>(loc, floatType, d1, sumiBlockFloat);
          mlir::Value sumiNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumiCur, term);
          rewriter.create<emitc::YieldOp>(loc, sumiNext);
        }
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, subExpr.getResult());
      }

      // sumf = sumf + d0 * sumi;  (the super-block term, ggml EXACT order, ONE
      // emitc.expression -> ONE C FMA).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiSuper =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, d0, sumiSuper);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    return sumfFinal;
  }

// The typed per-block dual-fp16 SCALE reconstruction primitive. This lowers the
// SAME two scalar fp16->fp32 reads + scalar float multiply the monolithic
// block-dot emitters produce inline for `d_x * d_y`, but as a first-class typed
// body op (I5) rather than an opaque inline lambda. The emitted C is byte-EQUAL
// to the monolith's inline `d_x * d_y` because kFp16ScaleReadCallee currently
// holds the same spelling as the monolith's independent local fp16ReadCallee
// literals (:187, :479, :5428, :6022) and the float emitc.mul spelling matches
// -- a byte-equal-literal coincidence, NOT a mechanized single-source share
// (those monolith literals do not reference this constant). The consolidation
// that actually mechanizes drift-protection is deferred to brick(5) (the q8_0
// wire-in), where the monolith inline reads are replaced by this typed op. The
// op is scalar (no vl); bodyVL is unused.
mlir::LogicalResult VariantToEmitCFunc::emitBlockFp16ScaleProduct(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::BlockFp16ScaleProductOp scaleProduct,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
  mlir::Type sizeType = getSizeType(rewriter);

  mlir::Value lhsBase = valueMap.lookup(scaleProduct.getLhsScaleBase());
  mlir::Value rhsBase = valueMap.lookup(scaleProduct.getRhsScaleBase());
  if (!lhsBase || !rhsBase)
    return rewriter.notifyMatchFailure(scaleProduct,
                                       "block_fp16_scale_product operand "
                                       "unmapped");

  llvm::StringRef opName = scaleProduct.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = scaleProduct.getWEFTEmitCLowerableSourceRole();

  // M-FLAT step 3 -- the loop-capable per-block-source form. When block_index
  // is present it is the enclosing weft_rvv.typed_flat_block_dot_loop_body
  // region's induction variable, so each per-block fp16 scale header lives at
  // `base + block_index*stride (+ byte_offset)`. This branch replicates
  // emitFlatBlockDot's blockBaseValue (:5451-5462, blockOffset 0) + fp16ReadAt
  // (:5467-5476) byte-exact: the SAME size_t emitc.mul + pointer emitc.add
  // block-base arithmetic and the SAME `(float)*(const _Float16 *)` call_opaque
  // read (spelled with fp16ReadAt's "fcvt.s.h" verbatim -- unlike the single-
  // block form below, which keeps its own frozen "(float)*(const _Float16 *)"
  // verbatim; only the per-block form is the byte-exact monolith replica). The
  // imported ABI bases stay the loop-invariant block-0 pointers; only the loop
  // offset is added. The single-block (block_index absent) path below is
  // untouched.
  if (mlir::Value blockIndex = scaleProduct.getBlockIndex()) {
    mlir::Value ib = valueMap.lookup(blockIndex);
    if (!ib)
      return rewriter.notifyMatchFailure(
          scaleProduct, "block_fp16_scale_product block_index unmapped");

    auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
    // const uint8_t *xb = base + ib*stride;  -- byte-exact to blockBaseValue
    // (blockOffset 0): a size_t emitc.mul then a pointer emitc.add.
    auto perBlockBase = [&](mlir::Value base, int64_t stride,
                            const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, base.getType(), base, off);
    };
    // (float)*(const _Float16 *)(blockBase (+ byte_offset))  -- byte-exact to
    // fp16ReadAt: the optional pointer emitc.add for the header byte offset then
    // the sanctioned opaque read.
    auto perBlockRead = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset && *byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(), blockBase,
                                             sizeLit(*byteOffset));
      return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                            mlir::ValueRange{addr}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };

    mlir::Value xb = perBlockBase(
        lhsBase, static_cast<int64_t>(*scaleProduct.getLhsBlockStride()),
        "block_base_x");
    mlir::Value yb = perBlockBase(
        rhsBase, static_cast<int64_t>(*scaleProduct.getRhsBlockStride()),
        "block_base_y");
    mlir::Value dX = perBlockRead(xb, scaleProduct.getLhsScaleByteOffset());
    mlir::Value dY = perBlockRead(yb, scaleProduct.getRhsScaleByteOffset());
    // float scale = d_x * d_y;  (ggml's q8_0 scale order: scales multiplied
    // FIRST)
    mlir::Value scale =
        rewriter.create<emitc::MulOp>(loc, floatType, dX, dY).getResult();
    valueMap[scaleProduct.getResult()] = scale;
    return mlir::success();
  }

  // Per-block fp16 read at `base (+ byte_offset)`. Default offset 0 reads the
  // AoS fp16 header at the block base (matching the monolithic q8_0 read).
  auto fp16Read = [&](mlir::Value base,
                      std::optional<int64_t> byteOffset) -> mlir::Value {
    mlir::Value readBase = base;
    if (byteOffset && *byteOffset != 0) {
      mlir::Value offset =
          rewriter.create<emitc::LiteralOp>(loc, sizeType,
                                            std::to_string(*byteOffset));
      readBase = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                               offset);
    }
    return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                          mlir::ValueRange{readBase}, opName, role);
  };

  mlir::Value dX = fp16Read(lhsBase, scaleProduct.getLhsScaleByteOffset());
  mlir::Value dY = fp16Read(rhsBase, scaleProduct.getRhsScaleByteOffset());
  // float scale = d_x * d_y;  (ggml's q8_0 scale order: scales multiplied FIRST)
  mlir::Value scale =
      rewriter.create<emitc::MulOp>(loc, floatType, dX, dY).getResult();
  valueMap[scaleProduct.getResult()] = scale;
  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitBlockComputedScaleDequant(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::BlockComputedScaleDequantOp dequant,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");

  mlir::Value sumi = valueMap.lookup(dequant.getSumi());
  mlir::Value scale = valueMap.lookup(dequant.getComputedScale());
  if (!sumi || !scale)
    return rewriter.notifyMatchFailure(dequant,
                                       "block_computed_scale_dequant operand "
                                       "unmapped");

  // float term = (float)sumi * scale;  -- byte-identical at the operation
  // spelling level to the monolithic block-dot fold: the SAME i32 -> float
  // emitc.cast (the `(float)sumi` sitofp) + the SAME scalar float emitc.mul the
  // monolith produces inline for the per-block `(float)sumi * <scale>` term.
  // scale is the COMPUTED weft_rvv.block_fp16_scale_product output (d_x*d_y),
  // not an imported ABI scale. This op stops at the per-block term; the
  // cross-block fp32 accumulate (`sumf += term`) is a separate typed step.
  mlir::Value sumiFloat =
      rewriter.create<emitc::CastOp>(loc, floatType, sumi).getResult();
  mlir::Value term =
      rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, scale)
          .getResult();
  valueMap[dequant.getResult()] = term;
  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitCrossBlockF32Accumulate(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::CrossBlockF32AccumulateOp accumulate,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");

  mlir::Value acc = valueMap.lookup(accumulate.getAcc());
  mlir::Value term = valueMap.lookup(accumulate.getTerm());
  if (!acc || !term)
    return rewriter.notifyMatchFailure(accumulate,
                                       "cross_block_f32_accumulate operand "
                                       "unmapped");

  // float sumf = acc + term;  -- byte-identical at the operation-spelling level
  // to the monolithic block-dot cross-block fold: the SAME scalar float
  // emitc.add the monolith produces for `sumf + <block term>`. The caller folds
  // in STRICT ascending block order (block-carried), so the fp non-associativity
  // matches ggml byte-for-byte. acc is the block-carried f32 accumulator, term
  // is the COMPUTED weft_rvv.block_computed_scale_dequant output
  // (`(float)sumi * scale`). The op stops at the fold; the block loop and the
  // final scalar store are separate typed steps.
  mlir::Value sumfNext =
      rewriter.create<emitc::AddOp>(loc, floatType, acc, term).getResult();
  valueMap[accumulate.getResult()] = sumfNext;
  return mlir::success();
}

mlir::LogicalResult VariantToEmitCFunc::emitTypedVectorLane0ToScalarExtract(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::TypedVectorLane0ToScalarExtractOp extract,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");

  mlir::Value input = valueMap.lookup(extract.getInput());
  if (!input)
    return rewriter.notifyMatchFailure(
        extract, "typed_vector_lane0_to_scalar_extract input unmapped");

  // int32_t sumi = __riscv_vmv_x_s_i32m1_i32(red);  -- byte-identical at the
  // operation-spelling level to the monolithic block-dot lane0 extraction
  // (emitFlatBlockDot:5653-5655): the SAME __riscv_vmv_x_s_i32m1_i32 call_opaque
  // that pulls the vwredsum lane0 into the scalar sumi. The intrinsic targets
  // lane 0 regardless of vl, so the call takes ONLY the i32m1 vector value (the
  // op's vl operand is the boundary marker, not a call argument).
  std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
  mlir::Value scalar =
      emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                     mlir::ValueRange{input},
                     extract.getWEFTEmitCLowerableSourceOpName(),
                     extract.getWEFTEmitCLowerableSourceRole());
  valueMap[extract.getResult()] = scalar;
  return mlir::success();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
