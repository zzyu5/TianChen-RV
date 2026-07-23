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

// Shared flat block-dot mechanical emission from a complete descriptor.

mlir::LogicalResult VariantToEmitCFunc::emitFlatBlockDot(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value output,
    mlir::Value blockDotResult, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, llvm::StringRef opName,
    llvm::StringRef role, const BlockDotFacts &facts,
    const FlatBlockDotDescriptor &descriptor) const {
  mlir::MLIRContext *ctx = rewriter.getContext();

  // The shared per-block emit state (interned types + wideLmul + bases). The
  // sumf accumulator lvalue and the codebook table register are filled in below
  // once their decls are emitted. buildFlatBlockDotEmitState is the SINGLE
  // source of the interned type spellings, so the monolithic path here and the
  // M-FLAT typed_flat_block_dot_loop_body region driver emit BYTE-IDENTICAL
  // per-block cores.
  FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
      rewriter, descriptor, facts, weightBase, activationBase,
      /*sumfVar=*/mlir::Value(), /*codebookValues=*/mlir::Value(), sizeType,
      opName, role);
  int64_t multiBlockFactor = facts.multiBlockFactor;
  int64_t qk = descriptor.qk;

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // The 16-entry codebook (2nd primitive class) is a STRUCTURAL fact off the
  // op's DenseI8ArrayAttr. Emit it as a `static const int8_t <name>[N]` decl
  // ONCE, before the accumulator -- the task-sanctioned structured const for the
  // gather table (the decl renders the verified attr entries; the register is
  // broadcast-loaded below the block count and reused by every gather).
  if (descriptor.hasCodebook) {
    std::string decl =
        ("static const int8_t " + descriptor.codebookTableName + "[" +
         std::to_string(descriptor.codebook.size()) + "] = {")
            .str();
    for (size_t i = 0; i < descriptor.codebook.size(); ++i) {
      if (i)
        decl += ", ";
      decl += std::to_string(static_cast<int>(descriptor.codebook[i]));
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);
  }

  // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
  rewriter.create<emitc::VerbatimOp>(
      loc, localVariableComment("sumf", opName, role));
  auto sumfVar = rewriter.create<emitc::VariableOp>(
      loc, emitc::LValueType::get(st.floatType),
      emitc::OpaqueAttr::get(ctx, ""));
  rewriter.create<emitc::AssignOp>(
      loc, sumfVar,
      rewriter.create<emitc::LiteralOp>(loc, st.floatType, "0.0f"));
  st.sumfVar = sumfVar.getResult();

  // size_t nb = n / QK;
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  // The codebook table broadcast into a vector register ONCE (reused by every
  // gather): vint8<L>_t values = vle8_v_i8<L>(<name>, N). The table pointer is
  // the structured-const decl above, spelled at the i8 anchor LMUL. Null for the
  // non-codebook primitives (the gather case is the only reader).
  if (descriptor.hasCodebook) {
    std::string tableLoadCallee = riscvIntrinsicName("vle", 8, st.coreLmul, "i8");
    st.codebookValues = emitOpaqueCallBuilt(
        rewriter, loc, st.i8CoreType, tableLoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value tableName = rewriter.create<emitc::LiteralOp>(
              loc, st.i8PtrType, descriptor.codebookTableName.str());
          return {tableName, sizeLit(descriptor.codebook.size())};
        },
        llvm::StringRef("codebook_table_load"));
  }

  if (multiBlockFactor == 1) {
    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the no-unroll form.
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());
    mlir::FailureOr<FlatBlockCore> core = emitFlatBlockCore(
        rewriter, loc, st, blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
    if (mlir::failed(core))
      return mlir::failure();
    emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY, core->mX,
                 core->sY);
  } else {
    // Multi-block unroll: a main loop stepping by factor over nb - nb%factor
    // full groups -- emit ALL factor independent cores FIRST (the latency-
    // overlap lever), THEN the factor folds in strict ascending block order --
    // then a robust single-block scalar tail over the nb % factor remainder.
    mlir::Value factorLit = sizeLit(multiBlockFactor);
    mlir::Value nbRem =
        rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
    mlir::Value nbMain =
        rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                  factorLit,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      llvm::SmallVector<FlatBlockCore> cores;
      for (int64_t k = 0; k < multiBlockFactor; ++k) {
        mlir::FailureOr<FlatBlockCore> core = emitFlatBlockCore(
            rewriter, loc, st, mainLoop.getInductionVar(), k,
            /*forceRobust=*/false);
        if (mlir::failed(core))
          return mlir::failure();
        cores.push_back(*core);
      }
      for (const FlatBlockCore &core : cores)
        emitFlatFold(rewriter, loc, st, core.sumiVar, core.dX, core.dY, core.mX,
                     core.sY);
    }
    auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      mlir::FailureOr<FlatBlockCore> core = emitFlatBlockCore(
          rewriter, loc, st, tailLoop.getInductionVar(), 0,
          /*forceRobust=*/true);
      if (mlir::failed(core))
        return mlir::failure();
      emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                   core->mX, core->sY);
    }
  }

  // *s = sumf;  (structured scalar store through the output pointer)
  auto outPointer =
      llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
  if (!outPointer)
    return rewriter.notifyMatchFailure(blockDotResult.getDefiningOp(),
                                       "block-dot output not a pointer");
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "store_s"));
  mlir::Value outIndex =
      rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
  emitc::SubscriptOp outSubscript =
      rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
  mlir::Value sumfFinal =
      rewriter.create<emitc::LoadOp>(loc, st.floatType, st.sumfVar).getResult();
  rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

  valueMap[blockDotResult] = sumfFinal;
  return mlir::success();
}

// Build the shared per-block emit state both flat-block-dot callers use. The
// interned emitc type spellings + the wideLmul superset formula are single-
// sourced here so the monolithic emitFlatBlockDot and the M-FLAT loop-body
// driver emit BYTE-IDENTICAL per-block cores. weightPtrType/activationPtrType
// come straight off the imported ABI base pointer types; sumfVar/codebookValues
// are filled in by the caller after their decls are emitted.
FlatBlockDotEmitState VariantToEmitCFunc::buildFlatBlockDotEmitState(
    mlir::ConversionPatternRewriter &rewriter,
    const FlatBlockDotDescriptor &descriptor, const BlockDotFacts &facts,
    mlir::Value weightBase, mlir::Value activationBase, mlir::Value sumfVar,
    mlir::Value codebookValues, mlir::Type sizeType, llvm::StringRef opName,
    llvm::StringRef role) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  FlatBlockDotEmitState st;
  st.descriptor = descriptor;
  st.opName = opName;
  st.role = role;
  st.coreLmul = facts.coreLmul;
  // i8 source LMUL -> the next-wider i16 product LMUL (m2->m4, m1->m2, mf2->m1,
  // mf4->mf2). Byte-exact for every in-tree anchor.
  st.wideLmul =
      (facts.coreLmul == "m2")    ? "m4"
      : (facts.coreLmul == "m1")  ? "m2"
      : (facts.coreLmul == "mf2") ? "m1"
                                  : "mf2";
  st.stripElided = facts.stripElided;
  st.weightBase = weightBase;
  st.activationBase = activationBase;
  st.sumfVar = sumfVar;
  st.codebookValues = codebookValues;
  st.sizeType = sizeType;
  st.floatType = emitc::OpaqueType::get(ctx, "float");
  st.i32Type = emitc::OpaqueType::get(ctx, "int32_t");
  st.u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
  st.weightPtrType = weightBase.getType();
  st.activationPtrType = activationBase.getType();
  std::string i8CoreTypeName = ("vint8" + facts.coreLmul + "_t").str();
  std::string u8CoreTypeName = ("vuint8" + facts.coreLmul + "_t").str();
  std::string i16WideTypeName = ("vint16" + st.wideLmul + "_t").str();
  std::string u16WideTypeName = ("vuint16" + st.wideLmul + "_t").str();
  st.i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
  st.u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
  st.i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
  st.u16WideType = emitc::OpaqueType::get(ctx, u16WideTypeName);
  st.i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
  st.i8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
  st.u8PtrType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
  st.fp16ReadCallee = "(float)*(const _Float16 *)";
  st.u16ReadCallee = "(uint16_t)*(const uint16_t *)";
  return st;
}

// Factored VERBATIM from emitFlatBlockDot's former emitIntegerCore + the strip
// reduce it drives; reads all shared state off `st`. Byte-identical emit.
mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitFlatIntegerCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const FlatBlockDotEmitState &st, mlir::Value xb, mlir::Value yb,
    mlir::Value qhLow16, mlir::Value qhHigh16, bool forceRobust) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  const FlatBlockDotDescriptor &descriptor = st.descriptor;
  llvm::StringRef opName = st.opName;
  llvm::StringRef role = st.role;
  llvm::StringRef coreLmul = st.coreLmul;
  llvm::StringRef wideLmul = st.wideLmul;
  bool stripElided = st.stripElided;
  int64_t blockLen = descriptor.blockLen;
  int64_t quantOffset = descriptor.quantOffset;
  int64_t actQuantOffset = descriptor.activationQuantOffset;
  int64_t highOffset = descriptor.highOffset;
  mlir::Type sizeType = st.sizeType;
  mlir::Type i32Type = st.i32Type;
  mlir::Type weightPtrType = st.weightPtrType;
  mlir::Type activationPtrType = st.activationPtrType;
  mlir::Type i8CoreType = st.i8CoreType;
  mlir::Type u8CoreType = st.u8CoreType;
  mlir::Type i16WideType = st.i16WideType;
  mlir::Type u16WideType = st.u16WideType;
  mlir::Type i32m1Type = st.i32m1Type;
  mlir::Type i8PtrType = st.i8PtrType;
  mlir::Type u8PtrType = st.u8PtrType;
  mlir::Value codebookValues = st.codebookValues;
  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

    // The decode+product for ONE strip -- the ONLY arithmetic divergence, a
    // switch over the EXISTING factored decode helpers -- followed by the SHARED
    // seed/vwredsum/extract tail. qhLow16/qhHigh16 are null for the non-five-bit
    // primitives. chunkOffset is the within-block byte offset (the strip
    // induction var for the robust loop, or a 0 literal for the elided core).
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value qhLow16, mlir::Value qhHigh16,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          mlir::Type castPtrType, int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castPtrType, full)
            .getResult();
      };
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      auto loadI8 = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      auto loadU8 = [&](mlir::Value ptr) -> mlir::Value {
        return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };

      mlir::FailureOr<mlir::Value> productOr = mlir::failure();
      switch (descriptor.decodePrimitive) {
      case FlatDecodePrimitive::PlainI8: {
        // Plain signed widening product: i8 x i8 -> i16 (NO nibble decode).
        mlir::Value vx0 =
            loadI8(chunkPtr(xb, weightPtrType, i8PtrType, quantOffset));
        mlir::Value vy0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        std::string mulCallee =
            riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
        productOr =
            emitOpaqueCall(rewriter, loc, i16WideType, mulCallee,
                           mlir::ValueRange{vx0, vy0, vl}, opName, role);
        break;
      }
      case FlatDecodePrimitive::OffsetBinaryNibble: {
        // Offset-binary asymmetric i4xi8 (weight loaded signed i8; xor 0x88 +
        // sll/sra sign-extend + vwmul/vwmacc against the low/high q8 halves).
        mlir::Value w =
            loadI8(chunkPtr(xb, weightPtrType, i8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        productOr = emitOffsetBinaryDecodeProductValue(
            rewriter, loc, w, y0, y1, vl, i8CoreType, i16WideType, "i8",
            coreLmul, 16, wideLmul, "i16", opName, role);
        break;
      }
      case FlatDecodePrimitive::UnsignedNibble: {
        // Unsigned-nibble asymmetric i4xi8 (weight loaded u8; vand 0x0F / vsrl
        // 0x04 + reinterpret to i8 + vwmul/vwmacc).
        mlir::Value w =
            loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        productOr = emitUnsignedNibbleDecodeProductValue(
            rewriter, loc, w, y0, y1, vl, i8CoreType, u8CoreType, i16WideType,
            coreLmul, 16, wideLmul, "i16", opName, role);
        break;
      }
      case FlatDecodePrimitive::FiveBitOffsetBinary: {
        // 5-bit offset-binary (weight loaded u8 at quantOffset; q8 halves at the
        // DISTINCT actQuantOffset; nibble + qh 5th-bit merge; applyOffsetBias
        // selects the `-16` for q5_0 vs the MIN-scale bias for q5_1).
        mlir::Value w =
            loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
        productOr = emitFiveBitOffsetBinaryDecodeProductValue(
            rewriter, loc, w, y0, y1, qhLow16, qhHigh16, chunkOffset, vl,
            i8CoreType, u8CoreType, u16WideType, i16WideType, coreLmul, wideLmul,
            16, wideLmul, "i16", opName, role, descriptor.applyOffsetBias);
        break;
      }
      case FlatDecodePrimitive::CodebookGatherNibble: {
        // 2nd primitive class: split the packed weight byte into the two UNSIGNED
        // nibble index lanes (vand 0x0F / vsrl 0x04), GATHER each through the
        // broadcast codebook table (vrgather_vv_i8<L>) into signed-i8 weight lanes
        // v0/v1, then feed the SAME asymmetric signed widening product the
        // offset-binary sibling uses (vwmul low <-> q8[0..15], vwmacc + high).
        mlir::Value w =
            loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
        mlir::Value y0 =
            loadI8(chunkPtr(yb, activationPtrType, i8PtrType, actQuantOffset));
        mlir::Value y1 = loadI8(chunkPtr(yb, activationPtrType, i8PtrType,
                                         actQuantOffset + highOffset));
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
                                mlir::ValueRange{codebookValues, idx, vl}, opName,
                                role);
        };
        mlir::Value v0 = gather(idxLow);
        mlir::Value v1 = gather(idxHigh);
        productOr = emitOffsetBinaryProductFromDecodedValue(
            rewriter, loc, v0, v1, y0, y1, vl, i16WideType, 16, wideLmul, "i16",
            opName, role);
        break;
      }
      case FlatDecodePrimitive::BinarySign:
      case FlatDecodePrimitive::NVFP4Codebook:
        llvm_unreachable(
            "closed whole-body flat families never enter shared strip emission");
      }
      if (mlir::failed(productOr))
        return mlir::failure();
      mlir::Value product = *productOr;

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, seedCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value sumiSeed =
                carrySumi
                    ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
            mlir::Value one = sizeLit(1);
            return {sumiSeed, one};
          });
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red =
          emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                         mlir::ValueRange{product, seed, vl}, opName, role);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      return emitOpaqueCall(rewriter, loc, i32Type, extractCallee,
                            mlir::ValueRange{red}, opName, role);
    };
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The vsetvl SEW/LMUL spelling of the anchor: m1/m2 use vsetvl_e8<lmul>
      // (byte strips), mf4 uses vsetvl_e32m1 (4-element strips at VLEN=128).
      unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
      llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");

      if (!forceRobust && stripElided) {
        // Elided core (VLEN >= guaranteed floor): ONE vsetvl(block_len) caps the
        // active vl at the whole (half-)block + ONE strip reduce. NO inner strip
        // loop, NO sumi carry (seed lane0 = 0).
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, qhLow16, qhHigh16, sizeLit(0), vl, sumiVar,
                            /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the block_len bytes; the loop STEP
      // is the loop-invariant VLMAX, the per-chunk active vl is vsetvl(len - c).
      // Stays VLEN-robust via the sumi-carrying seed.
      mlir::Value innerVlmax = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(blockLen)};
          });
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value remaining = rewriter.create<emitc::SubOp>(
                  loc, sizeType, sizeLit(blockLen), c);
              return {remaining};
            });

        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, qhLow16, qhHigh16, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
}

// Factored VERBATIM from emitFlatBlockDot's former emitFold; reads shared state
// off `st`. The fold tree switches on st.descriptor.foldModel and folds the
// dX/dY (+ mX/sY) OPERANDS -- region-driven, not attribute-rederived.
void VariantToEmitCFunc::emitFlatFold(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const FlatBlockDotEmitState &st, mlir::Value sumiVar, mlir::Value dX,
    mlir::Value dY, mlir::Value mX, mlir::Value sY) const {
  const FlatBlockDotDescriptor &descriptor = st.descriptor;
  llvm::StringRef opName = st.opName;
  llvm::StringRef role = st.role;
  mlir::Type floatType = st.floatType;
  mlir::Type i32Type = st.i32Type;
  mlir::Value sumfVar = st.sumfVar;

      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();

      // Pinned fp-fold oracle [K-5]/[measurement/浮点折叠oracle.md §1]:
      // strict left-assoc, ordered, NO dx*dy premultiply, NO FMA contraction.
      // Emitted as SEPARATE emitc statements (standalone cast/mul/mul/add, NOT
      // inside one emitc.expression) so clang's default -ffp-contract=on cannot
      // fuse the (t*d_y)+sumf into an fmaf -- cross-statement contraction with
      // named intermediates is not permitted. The two muls give ((sumi*d_x)*d_y),
      // the add folds into the ordered running sumf:
      //   float t = (float)sumi * d_x;   // t = f32(sumi) (x) dx
      //   t = t * d_y;                   // t = t (x) dy
      //   sumf = sumf + t;               // ordered (+)
      if (descriptor.foldModel == FlatFoldModel::SeparatedLeftAssoc) {
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value t =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
        mlir::Value t2 = rewriter.create<emitc::MulOp>(loc, floatType, t, dY);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, t2);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumf", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfNext);
        return;
      }

      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumfNext;
        switch (descriptor.foldModel) {
        case FlatFoldModel::SumiTimesScales: {
          // sumf + (float)sumi * (d_x * d_y)  (ggml q8_0 order: scales FIRST).
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value scaleProduct =
              rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
          mlir::Value blockTerm = rewriter.create<emitc::MulOp>(
              loc, floatType, sumiFloat, scaleProduct);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::LeftAssoc: {
          // sumf + ((float)sumi * d_x) * d_y  (ggml q4_0 left-assoc order).
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value timesDx =
              rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
          mlir::Value blockTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::ScalesTimesSumi: {
          // sumf + (d_x * d_y) * (float)sumi  (ggml q5_0 order: scales FIRST as
          // the LEFT operand -- distinct emitc sequence from q8_0's above).
          mlir::Value scales =
              rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value blockTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, scales, sumiFloat);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::ScalePlusMin: {
          // sumf + ((d_x*d_y)*sumi + m_x*s_y)  (ggml q4_1/q5_1 exact tree: the
          // two products are SUMMED FIRST, then added to sumf).
          mlir::Value sumiFloat =
              rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                  .getResult();
          mlir::Value scaleProduct =
              rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
          mlir::Value scaleTerm = rewriter.create<emitc::MulOp>(
              loc, floatType, scaleProduct, sumiFloat);
          mlir::Value minTerm =
              rewriter.create<emitc::MulOp>(loc, floatType, mX, sY);
          mlir::Value blockTerm =
              rewriter.create<emitc::AddOp>(loc, floatType, scaleTerm, minTerm);
          sumfNext =
              rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
          break;
        }
        case FlatFoldModel::SeparatedLeftAssoc:
          // Handled above via SEPARATE emitc statements (no fused expression),
          // so it never reaches this expression-body switch.
          llvm_unreachable(
              "SeparatedLeftAssoc is emitted before the fused-expression switch");
        case FlatFoldModel::BinaryTwoLevel:
        case FlatFoldModel::NVFP4Codebook:
          llvm_unreachable(
              "closed whole-body flat families never enter shared fold emission");
        }
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
}

// Factored VERBATIM from emitFlatBlockDot's former emitBlockCore + its per-block
// address / fp16-read / qh helpers; reads shared state off `st`. Byte-identical
// emit. The integer core is driven through the SHARED emitFlatIntegerCore.
mlir::FailureOr<FlatBlockCore> VariantToEmitCFunc::emitFlatBlockCore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const FlatBlockDotEmitState &st, mlir::Value ib, int64_t blockOffset,
    bool forceRobust) const {
  const FlatBlockDotDescriptor &descriptor = st.descriptor;
  llvm::StringRef opName = st.opName;
  llvm::StringRef role = st.role;
  mlir::Type sizeType = st.sizeType;
  mlir::Type floatType = st.floatType;
  mlir::Type u32Type = st.u32Type;
  mlir::Type weightPtrType = st.weightPtrType;
  mlir::Type activationPtrType = st.activationPtrType;
  mlir::Value weightBase = st.weightBase;
  mlir::Value activationBase = st.activationBase;
  int64_t weightStride = descriptor.weightStride;
  int64_t activationStride = descriptor.activationStride;
  llvm::StringRef fp16ReadCallee = st.fp16ReadCallee;
  llvm::StringRef u16ReadCallee = st.u16ReadCallee;
  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*Sw;
    // const uint8_t *yb = vy + (ib+blockOffset)*Sa.
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };
    // A scalar fp16->fp32 read at a per-block byte offset (the sanctioned opaque
    // piece). At byteOffset 0 no emitc.add is emitted, so this is byte-identical
    // to the family-A `fp16Read(base)` form and to the family-B `fp16ReadAt`.
    auto fp16ReadAt = [&](mlir::Value blockBase, mlir::Type ptrType,
                          int64_t byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, ptrType, blockBase,
                                             sizeLit(byteOffset));
      return emitOpaqueCall(rewriter, loc, floatType, fp16ReadCallee,
                            mlir::ValueRange{addr}, opName, role,
                            llvm::StringRef("fcvt.s.h"));
    };
    // The per-element 5th-bit qh field (five-bit primitive only), read as two
    // ALIGNED 16-bit halves (LE (qh & 0xFFFF) / (qh >> 16)).
    struct QhHalves {
      mlir::Value low16;
      mlir::Value high16;
    };
    auto qhRead = [&](mlir::Value xb) -> QhHalves {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qh_field"));
      auto readHalf = [&](int64_t byteOffset) -> mlir::Value {
        mlir::Value ptr = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, xb, sizeLit(byteOffset));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                         u16ReadCallee, mlir::ValueRange{ptr})
            .getResult(0);
      };
      mlir::Value low16 = readHalf(descriptor.qhOffset);
      mlir::Value high16 = readHalf(descriptor.qhOffset + 2);
      return QhHalves{low16, high16};
    };
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      // The weight scale: the sanctioned fp16 read (every flat-plain + iq4_nl
      // format) or the structured E8M0 -> fp32 half reconstruction (mxfp4, no
      // fp16 weight field). The activation scale is always the fp16 read.
      mlir::Value dX =
          (descriptor.weightScaleSource == FlatWeightScaleSource::E8M0)
              ? emitE8M0HalfScale(rewriter, loc, xb, opName, role)
              : fp16ReadAt(xb, weightPtrType, 0);
      mlir::Value dY = fp16ReadAt(yb, activationPtrType, 0);
      mlir::Value mX = nullptr;
      mlir::Value sY = nullptr;
      if (descriptor.hasMinTerm) {
        mX = fp16ReadAt(xb, weightPtrType, descriptor.weightMinOffset);
        sY = fp16ReadAt(yb, activationPtrType, descriptor.activationSumOffset);
      }
      mlir::Value qhLow16 = nullptr;
      mlir::Value qhHigh16 = nullptr;
      if (descriptor.hasQh) {
        QhHalves qh = qhRead(xb);
        qhLow16 = qh.low16;
        qhHigh16 = qh.high16;
      }
      mlir::FailureOr<mlir::Value> sumiVar =
          emitFlatIntegerCore(rewriter, loc, st, xb, yb, qhLow16, qhHigh16, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return FlatBlockCore{*sumiVar, dX, dY, mX, sY};
}


// Lower a formula-constructed, complete typed flat block-dot body. The
// loop-carried f32 accumulator is projected to the EmitC mutable variable, while
// every compute-family choice comes from the formula-produced flat_* plan and
// every integer/fold mechanism is wired from the typed region. Incomplete
// mechanism shells are rejected before this function is entered as a production
// fallback.

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
