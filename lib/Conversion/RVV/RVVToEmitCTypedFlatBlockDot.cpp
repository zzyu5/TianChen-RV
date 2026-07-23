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

// Typed flat block-dot loop consumer; final flat_* plan is authoritative.

mlir::LogicalResult VariantToEmitCFunc::emitTypedFlatBlockDotLoopBody(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");

  weftrvv::TypedFlatBlockDotLoopBodyOp loopBody;
  for (mlir::Operation &op : scope.getBody().front()) {
    if (auto lb = llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(op))
      loopBody = lb;
  }
  if (!loopBody)
    return rewriter.notifyMatchFailure(
        scope, "typed flat block-dot loop body missing the op");

  mlir::Value weightBase = valueMap.lookup(loopBody.getWeightBase());
  mlir::Value activationBase = valueMap.lookup(loopBody.getActivationBase());
  mlir::Value output = valueMap.lookup(loopBody.getOutput());
  if (!weightBase || !activationBase || !output)
    return rewriter.notifyMatchFailure(loopBody,
                                       "loop-body ABI operand unmapped");

  llvm::StringRef opName = loopBody.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef role = loopBody.getWEFTEmitCLowerableSourceRole();
  int64_t qk = loopBody.getQk();
  std::optional<FlatBlockDotComputePlan> flatPlan =
      readFinalFlatBlockDotComputePlan(loopBody.getOperation());
  if (!flatPlan)
    return rewriter.notifyMatchFailure(
        loopBody, "flat block-dot body reached emission without a complete "
                  "formula-produced flat_* computation plan");

  auto sizeLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, sizeType, v); };
  auto matchesFlatPlanSingleActivationOffset =
      [&](weftrvv::LoadOp load) {
        return load && load.getQuantByteOffset() &&
               static_cast<int64_t>(*load.getQuantByteOffset()) ==
                   flatPlan->activationQuantOffset;
      };
  auto matchesFlatPlanHalfActivationOffsets =
      [&](weftrvv::LoadOp low, weftrvv::LoadOp high) {
        return low && high && low.getQuantByteOffset() &&
               high.getQuantByteOffset() &&
               static_cast<int64_t>(*low.getQuantByteOffset()) ==
                   flatPlan->activationQuantOffset &&
               static_cast<int64_t>(*high.getQuantByteOffset()) ==
                   flatPlan->activationQuantOffset + flatPlan->blockLen;
      };

  // ===================================================================
  // q1_0 (flat_binary_two_level) BINARY-sign full-body emit. The q1_0
  // per-super-block contribution is a FOUR-sub-block binary sign decode with a
  // distinct TWO-LEVEL fp32 fold (`d0 * Σ_k(d1_k * sumi_block_k)`) that no
  // existing single-core flat brick chain expresses, so the WHOLE per-super-block
  // body (including the fold) is carried by ONE net-new binary-sign integer-core
  // brick + re-emitted here -- the super-block scalar-core precedent (tq1_0/iq1_s)
  // applied to the flat scaffold. The typed path preserves the previously sealed
  // byte-exact output while owning the only live emitter. Handled BEFORE the
  // shared sumf/nb prelude below
  // (the shared body emits its own), so it returns here. Anti-bypass (I7): the
  // ABI bases are sourced from the BRICK's operands (not the loop op attrs) and
  // the brick's block_index MUST be the region induction variable.
  if (flatPlan->bodyFamily == FlatBodyFamily::BinaryTwoLevel) {
    weftrvv::GgmlBlockDotQ10Q80BinarySignCoreOp coreOp;
    weftrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotQ10Q80BinarySignCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
    });
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q1_0 flat_binary_two_level body requires the q1_0 "
                    "binary-sign integer-core brick + the loop yield");
    if (flatPlan->blockLen != static_cast<int64_t>(coreOp.getQk()) ||
        flatPlan->activationQuantOffset !=
            static_cast<int64_t>(coreOp.getActivationQuantByteOffset()))
      return rewriter.notifyMatchFailure(
          loopBody, "q1_0 formula plan geometry does not match the typed "
                    "binary-sign core");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "q1_0 flat_binary_two_level body region must carry exactly "
                    "the (block_index, sumf) pair");
    mlir::Value blockIndex = coreBlock.getArgument(0);
    mlir::Value accArg = coreBlock.getArgument(1);
    // The emit re-creates the whole body (including the two-level fold), so the
    // loop yield carries the loop-carried acc UNCHANGED (byte-exact = the monolith
    // sumf; the SSA fold is emitter-inlined, mirroring the tq1_0 scalar core).
    if (yieldOp.getAccNext() != accArg)
      return rewriter.notifyMatchFailure(
          yieldOp, "q1_0 flat_binary_two_level yield must carry the loop-carried "
                   "acc (the two-level fold is emitter-inlined by the binary-sign "
                   "brick lowering)");
    if (coreOp.getBlockIndex() != blockIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the q1_0 binary-sign core brick's block_index must be the "
                    "loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    mlir::Value weightBase = valueMap.lookup(coreOp.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(coreOp.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody, "q1_0 flat_binary_two_level ABI operand unmapped");
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(loopBody,
                                         "q1_0 loop-body output not a pointer");

    if (!coreOp.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          coreOp, "q1_0 core reached emission without final integer_core_lmul");
    llvm::StringRef coreLmul = *coreOp.getIntegerCoreLmul();

    (void)emitQ1_0TypedFlatBlockDotBody(
        rewriter, loc, weightBase, activationBase, outPointer, avlArg, sizeType,
        opName, role, coreLmul, flatPlan->blockLen,
        coreOp.getWeightBlockStride(),
        coreOp.getActivationBlockStride(),
        coreOp.getActivationBlocksPerWeight(),
        coreOp.getWeightQuantByteOffset(),
        flatPlan->activationQuantOffset);
    return mlir::success();
  }

  // ===================================================================
  // nvfp4 (flat_nvfp4_codebook) FP4-CODEBOOK full-body emit. nvfp4 (NVIDIA's FP4,
  // the SECOND FP4-class sibling) is a SUPER-BLOCK codebook quant whose 64 elements
  // span TWO block_q8_0 activation blocks -- a q8_0 activation STREAM (like q1_0),
  // so it rides the FLAT loop op (NOT the q8_K super-block one). Its per-super-block
  // body (the four 16-element sub-blocks, each with its UE4M3 weight scale + q8
  // block/half selection + the FP4-codebook gather integer core + the per-sub-block
  // fp32 fold `sumf += (d_y*d_x)*(float)sumi_s`) is carried WHOLE by ONE net-new
  // codebook integer-core brick + re-emitted here through emitNVFP4BlockDotBodyShared
  // -- the SAME shared body the retired monolith GgmlBlockDotNVFP4Q80Op emitter
  // called, so the emit is BYTE-EXACT to the monolith modulo only the source-op
  // provenance token. Handled BEFORE the shared sumf/nb prelude below (the shared
  // body emits its own), so it returns here. Anti-bypass (I7): the ABI bases are
  // sourced from the BRICK's operands (not the loop op attrs) and the brick's
  // block_index MUST be the region induction variable.
  if (flatPlan->bodyFamily == FlatBodyFamily::NVFP4Codebook) {
    weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp coreOp;
    weftrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o =
              llvm::dyn_cast<weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp>(bodyOp))
        coreOp = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
    });
    mlir::Block &coreBlock = loopBody.getBody().front();
    if (!coreOp || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "nvfp4 flat_nvfp4_codebook body requires the nvfp4 "
                    "codebook integer-core brick + the loop yield");
    if (flatPlan->blockLen != static_cast<int64_t>(coreOp.getQkSub()) ||
        flatPlan->activationQuantOffset !=
            static_cast<int64_t>(coreOp.getActivationQuantByteOffset()))
      return rewriter.notifyMatchFailure(
          loopBody, "nvfp4 formula plan geometry does not match the typed "
                    "codebook core");
    if (coreBlock.getNumArguments() != 2)
      return rewriter.notifyMatchFailure(
          loopBody, "nvfp4 flat_nvfp4_codebook body region must carry exactly "
                    "the (block_index, sumf) pair");
    mlir::Value blockIndex = coreBlock.getArgument(0);
    mlir::Value accArg = coreBlock.getArgument(1);
    // The emit re-creates the whole body (including the per-sub-block fold), so the
    // loop yield carries the loop-carried acc UNCHANGED (byte-exact = the monolith
    // sumf; the SSA fold is emitter-inlined, mirroring the q1_0 codebook core).
    if (yieldOp.getAccNext() != accArg)
      return rewriter.notifyMatchFailure(
          yieldOp, "nvfp4 flat_nvfp4_codebook yield must carry the loop-carried "
                   "acc (the per-sub-block fold is emitter-inlined by the "
                   "nvfp4 codebook brick lowering)");
    if (coreOp.getBlockIndex() != blockIndex)
      return rewriter.notifyMatchFailure(
          loopBody, "the nvfp4 codebook core brick's block_index must be the "
                    "loop induction variable (region arg 0) so the emit "
                    "addresses base + ib*stride, not super-block-0");

    mlir::Value weightBase = valueMap.lookup(coreOp.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(coreOp.getActivationBase());
    mlir::Value output = valueMap.lookup(loopBody.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(
          loopBody, "nvfp4 flat_nvfp4_codebook ABI operand unmapped");
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(loopBody,
                                         "nvfp4 loop-body output not a pointer");

    if (!coreOp.getIntegerCoreLmul())
      return rewriter.notifyMatchFailure(
          coreOp, "nvfp4 codebook core reached emission without its final m1 "
                  "integer_core_lmul construction fact");
    llvm::StringRef coreLmul = *coreOp.getIntegerCoreLmul();

    (void)emitNVFP4BlockDotBodyShared(
        rewriter, loc, weightBase, activationBase, outPointer, avlArg, sizeType,
        opName, role, coreLmul, coreOp.getQk(), flatPlan->blockLen,
        coreOp.getWeightBlockStride(), coreOp.getActivationBlockStride(),
        coreOp.getWeightQuantByteOffset(),
        flatPlan->activationQuantOffset,
        coreOp.getActivationHighByteOffset(), coreOp.getCodebook());
    return mlir::success();
  }

  if (flatPlan->bodyFamily != FlatBodyFamily::Shared)
    return rewriter.notifyMatchFailure(
        loopBody, "flat_* body family is not implemented by this typed loop "
                  "realizer");

  // All remaining flat-loop families consume a complete construction-time
  // schedule.  The two closed whole-body families above do not use these axes.
  std::optional<BlockDotFacts> finalSchedule =
      readFinalBlockDotFacts(loopBody);
  if (!finalSchedule)
    return rewriter.notifyMatchFailure(
        loopBody, "flat block-dot body reached emission without the complete "
                  "integer_core_lmul/multi_block_factor/strip_elision plan");
  int64_t multiBlockFactor = finalSchedule->multiBlockFactor;
  bool stripElided = finalSchedule->stripElided;

  // iq4_nl / FP4 codebook class (2nd primitive class): peek the region for the
  // 16-entry codebook table broadcast + the codebook-gather integer core. When
  // present, the codebook decl + broadcast are emitted at the SAME positions the
  // monolithic emitFlatBlockDot uses (the decl BEFORE the sumf accumulator; the
  // broadcast AFTER the block count, above the block loop), and the full-body
  // dispatch below routes to the codebook branch. Null for every non-codebook
  // fold (monotonic: their region carries no codebook brick), so those paths are
  // structurally unchanged.
  weftrvv::CodebookTableBroadcastOp peekCodebookTable;
  loopBody.getBody().walk(
      [&](weftrvv::CodebookTableBroadcastOp o) { peekCodebookTable = o; });
  weftrvv::CodebookGatherXI8ProductOp peekCodebookGather;
  loopBody.getBody().walk(
      [&](weftrvv::CodebookGatherXI8ProductOp o) { peekCodebookGather = o; });
  const bool planUsesCodebook =
      flatPlan->decodePrimitive == FlatDecodePrimitive::CodebookGatherNibble;
  if (planUsesCodebook != static_cast<bool>(peekCodebookTable) ||
      planUsesCodebook != static_cast<bool>(peekCodebookGather))
    return rewriter.notifyMatchFailure(
        loopBody, "formula-produced codebook plan does not match the typed "
                  "mechanism body");
  if (peekCodebookTable &&
      peekCodebookTable.getTableSymbol() != flatPlan->codebookTableName)
    return rewriter.notifyMatchFailure(
        loopBody, "formula-produced codebook table name does not match the "
                  "typed table-broadcast mechanism");

  rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

  // The 16-entry codebook (2nd primitive class) as a `static const int8_t
  // <table_symbol>[N]` decl ONCE, BEFORE the accumulator -- byte-exact to
  // emitFlatBlockDot:5434-5445. The decl renders the verified table-broadcast op's
  // codebook attr entries; the broadcast register is loaded below the block count.
  if (peekCodebookTable) {
    std::string decl =
        ("static const int8_t " + peekCodebookTable.getTableSymbol() + "[" +
         std::to_string(peekCodebookTable.getCodebook().size()) + "] = {")
            .str();
    for (size_t i = 0; i < peekCodebookTable.getCodebook().size(); ++i) {
      if (i)
        decl += ", ";
      decl +=
          std::to_string(static_cast<int>(peekCodebookTable.getCodebook()[i]));
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);
  }

  // float sumf = 0.0f;  -- byte-exact to emitFlatBlockDot's sumf accumulator
  // decl (:5410-5417). The SSA loop-carried acc lowers to this mutable
  // emitc.variable lvalue.
  rewriter.create<emitc::VerbatimOp>(
      loc, localVariableComment("sumf", opName, role));
  auto sumfVar = rewriter.create<emitc::VariableOp>(
      loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
  rewriter.create<emitc::AssignOp>(
      loc, sumfVar,
      rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

  // size_t nb = n / QK;  -- byte-exact to emitFlatBlockDot:5419-5422. n is the
  // scope AVL, exactly as the monolithic body derives it.
  rewriter.create<emitc::VerbatimOp>(
      loc, stepComment(opName, role, "block_count"));
  mlir::Value nb =
      rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

  // The codebook table broadcast into a vector register ONCE (reused by every
  // gather), AFTER the block count / above the block loop -- byte-exact to
  // emitFlatBlockDot:5468-5478. codebookValues threads the broadcast register
  // into the codebook branch's emit state. Null for the non-codebook folds (the
  // gather branch is the only reader).
  mlir::Value codebookValues;
  if (peekCodebookTable) {
    llvm::StringRef codebookCoreLmul = finalSchedule->coreLmul;
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + codebookCoreLmul + "_t").str());
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    std::string tableLoadCallee =
        riscvIntrinsicName("vle", 8, codebookCoreLmul, "i8");
    codebookValues = emitOpaqueCallBuilt(
        rewriter, loc, i8CoreType, tableLoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value tableName = rewriter.create<emitc::LiteralOp>(
              loc, i8PtrType, peekCodebookTable.getTableSymbol().str());
          return {tableName,
                  sizeLit(peekCodebookTable.getCodebook().size())};
        },
        llvm::StringRef("codebook_table_load"));
  }

  // Inspect the typed mechanism region WITHOUT emitting: this only checks that
  // the formula-produced flat_* plan and its mechanism body agree; it is not a
  // format/kind decision surface. A full body carries brick 2 (the
  // computed-scale dequant); the offset-binary-nibble + left-associative plan is
  // the one the schedule-parametrization step materializes across all legal knob
  // combinations.
  weftrvv::BlockComputedScaleDequantOp peekBrick2;
  loopBody.getBody().walk(
      [&](weftrvv::BlockComputedScaleDequantOp o) { peekBrick2 = o; });
  const bool isQ40ScheduleParam =
      peekBrick2 && flatPlan->bodyFamily == FlatBodyFamily::Shared &&
      flatPlan->decodePrimitive == FlatDecodePrimitive::OffsetBinaryNibble;
  // The q8_0 (sumi_times_scales) full body is the SECOND flat fold whose FULL
  // legal {integer_core_lmul, multi_block_factor, strip_elision} cross product is
  // schedule-parameterized (ported from the q4_0 scaffold). q8_0's native anchor
  // is the whole-block plain-i8 m2 core (blockLen = qk, two i8 loads, direct
  // vwmul), so its cross product is {m2}×mbf{1,2,4}×strip{robust,elided}.
  // The iq4_nl codebook body ALSO stamps fold_model "sumi_times_scales" (its fold
  // is the SumiTimesScales `(float)sumi * (d_x*d_y)` tree, node-identical to
  // q8_0's), so exclude it here by the codebook-gather region fact -- it takes the
  // dedicated codebook branch in the full-body dispatch below (q8_0's whole-block
  // plain-i8 schedule-param path carries no codebook brick, so it is unchanged).
  const bool isQ80ScheduleParam =
      peekBrick2 && flatPlan->bodyFamily == FlatBodyFamily::Shared &&
      flatPlan->decodePrimitive == FlatDecodePrimitive::PlainI8 &&
      !peekCodebookGather;

  // M-FLAT P2c: the deferred-ordered fold_structure (vector-batched seed-ordered
  // vfredosum.vs cross-block fold) is currently materialized ONLY for the q8_0
  // sumi_times_scales flat body. Fail-closed (I7): a deferred-ordered request
  // that would fall to the q4_0 branch or any body without
  // the full q8_0 integer core must NOT silently emit the per-block schedule
  // (IR-says-deferred / emit-does-per-block is a lie) -- reject it here.
  if (!loopBody.getFoldStructure())
    return rewriter.notifyMatchFailure(
        loopBody, "flat block-dot body reached emission without final "
                  "fold_structure");
  llvm::StringRef foldStructure = *loopBody.getFoldStructure();
  if (foldStructure == "deferred-ordered" && !isQ80ScheduleParam)
    return rewriter.notifyMatchFailure(
        loopBody,
        "deferred-ordered fold_structure is currently materialized only for the "
        "q8_0 (sumi_times_scales) full flat block-dot body; the other folds "
        "require the per-block default for the currently materialized fold");

  // [GAP-NUM] the numerics tier (which fp oracle governs the fold). "strict"
  // (default; absent = strict, fail-closed I7) issues the §1 byte-exact fold;
  // "relaxed" is the §5 policy-gated reassociation variant. The relaxed body is
  // currently materialized ONLY on the q8_0 deferred-ordered path (the vectorized
  // cross-block fold, where premultiply + vfmacc + one deferred vfredusum is the
  // minimal surgical delta). A relaxed request on any OTHER path must NOT silently
  // emit the strict fold (IR-says-relaxed / emit-does-strict is a lie) -- reject it
  // fail-closed so the tier is only ever honored where a relaxed body exists.
  if (!loopBody.getNumericsTier())
    return rewriter.notifyMatchFailure(
        loopBody, "flat block-dot body reached emission without final "
                  "numerics_tier");
  llvm::StringRef numericsTier = *loopBody.getNumericsTier();
  if (numericsTier == "relaxed" &&
      !(foldStructure == "deferred-ordered" && isQ80ScheduleParam))
    return rewriter.notifyMatchFailure(
        loopBody,
        "numerics_tier \"relaxed\" is currently materialized only for the q8_0 "
        "(sumi_times_scales) deferred-ordered flat block-dot body; every other "
        "path must use the strict tier on this materialized body");

  if (isQ40ScheduleParam) {
    // ===================================================================
    // q4_0 (left_assoc) SCHEDULE-PARAMETERIZED emit: the FULL legal
    // {integer_core_lmul, multi_block_factor, strip_elision} cross product,
    // driven by the loop-body knobs but sourced OP-BY-OP from the region ops
    // (the brick chain drives WHAT computes; the knobs drive HOW it schedules).
    // The schedule scaffold mirrors the monolithic emitFlatBlockDot (mbf-unroll
    // main loop + strict-ascending folds + robust tail; the inner strip loop for
    // the robust form), so each legal combo is byte-identical to the monolith's
    // same-knob GgmlBlockDotQ40Op instance while the integer core stays
    // region-driven (anti-bypass W4). ===================================
    mlir::Block &coreBlock = loopBody.getBody().front();

    // ---- Region walk (identify, no emit) ----
    weftrvv::BlockFp16ScaleProductOp brick1;
    weftrvv::BlockComputedScaleDequantOp brick2 = peekBrick2;
    weftrvv::CrossBlockF32AccumulateOp brick3;
    weftrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    weftrvv::StandaloneReduceOp coreReduce;
    weftrvv::TypedVectorLane0ToScalarExtractOp coreExtract;
    weftrvv::PackedI4OffsetBinaryXI8ProductOp packedProduct;
    llvm::SmallVector<weftrvv::LoadOp, 3> coreLoads;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::BlockFp16ScaleProductOp>(bodyOp))
        brick1 = o;
      else if (auto o = llvm::dyn_cast<weftrvv::CrossBlockF32AccumulateOp>(bodyOp))
        brick3 = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::StandaloneReduceOp>(bodyOp))
        coreReduce = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedVectorLane0ToScalarExtractOp>(
                       bodyOp))
        coreExtract = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::PackedI4OffsetBinaryXI8ProductOp>(
                       bodyOp))
        packedProduct = o;
      else if (auto o = llvm::dyn_cast<weftrvv::LoadOp>(bodyOp))
        coreLoads.push_back(o);
    });

    // ---- Region-driven gate (fail-closed, I7): the same brick-wiring checks the
    // mbf==1 path runs, so the schedule-parameterized emit provably tracks the
    // region content, not the attrs. ----
    if (!brick1 || !brick3 || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q4_0 schedule-parameterized body requires brick 1, brick 2, "
                    "brick 3, and the yield");
    if (brick1.getBlockIndex() != coreBlock.getArgument(0))
      return rewriter.notifyMatchFailure(
          brick1, "brick 1 block_index must be the loop induction variable");
    if (brick2.getComputedScale() != brick1.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 computed_scale must be brick 1's per-block scale");
    if (brick3.getAcc() != coreBlock.getArgument(1))
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 acc must be the loop-carried accumulator");
    if (brick3.getTerm() != brick2.getResult())
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 term must be brick 2's per-block dequant term");
    if (yieldOp.getAccNext() != brick3.getResult())
      return rewriter.notifyMatchFailure(
          yieldOp, "loop yield acc_next must be brick 3's fold result");
    if (!packedProduct || coreLoads.size() != 3 || !coreExtract)
      return rewriter.notifyMatchFailure(
          loopBody,
          "full q4_0 flat block-dot body requires the region integer core: three "
          "per-block i8 loads (a packed-i4 weight + two plain-i8 q8 halves), an "
          "asymmetric offset-binary packed-i4 x i8 product, and a lane0 scalar "
          "extract");
    if (brick2.getSumi() != coreExtract.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 sumi must be the integer-core lane0 extract result");
    auto q40WeightLoad = packedProduct.getWeight().getDefiningOp<weftrvv::LoadOp>();
    auto q40LowLoad =
        packedProduct.getActivationLow().getDefiningOp<weftrvv::LoadOp>();
    auto q40HighLoad =
        packedProduct.getActivationHigh().getDefiningOp<weftrvv::LoadOp>();
    if (!q40WeightLoad || !q40LowLoad || !q40HighLoad ||
        q40WeightLoad.getBuffer() != loopBody.getWeightBase() ||
        q40LowLoad.getBuffer() != loopBody.getActivationBase() ||
        q40HighLoad.getBuffer() != loopBody.getActivationBase() ||
        !q40WeightLoad.getQuantByteOffset() || !q40LowLoad.getQuantByteOffset() ||
        !q40HighLoad.getQuantByteOffset() || !q40WeightLoad.getBlockStride() ||
        !q40LowLoad.getBlockStride() ||
        !matchesFlatPlanHalfActivationOffsets(q40LowLoad, q40HighLoad))
      return rewriter.notifyMatchFailure(
          loopBody, "q4_0 packed-i4 product operands must be the region's "
                    "per-block loads off the ABI buffers with block_stride + "
                    "quant_byte_offset");
    if (!coreReduce || coreReduce.getInput() != packedProduct.getResult())
      return rewriter.notifyMatchFailure(
          coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
          "q4_0 integer-core reduce input must be the packed-i4 product");
    if (coreExtract.getInput() != coreReduce.getResult())
      return rewriter.notifyMatchFailure(
          coreExtract, "q4_0 integer-core lane0 extract input must be the reduce");

    // ---- The LeftAssoc fold descriptor + shared emit state (fold only; the
    // integer core is emitted op-by-op below). ----
    FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
    BlockDotFacts facts = *finalSchedule;
    FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
        rewriter, descriptor, facts, weightBase, activationBase,
        sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType, opName,
        role);

    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    llvm::StringRef coreLmul =
        llvm::cast<weftrvv::VectorType>(q40WeightLoad.getLoaded().getType())
            .getLmul();
    llvm::StringRef wideLmul =
        llvm::cast<weftrvv::VectorType>(packedProduct.getResult().getType())
            .getLmul();
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
    llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
    std::string innerSetvlCallee = riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
    int64_t blockLen = flatPlan->blockLen;

    // ---- The per-block q4_0 core, parameterized by (ib, blockOffset,
    // forceRobust). Emits address / brick-1 scales / the op-by-op integer core
    // (elided single cover OR robust inner strip loop) and returns the fold
    // inputs; the FOLD is emitted by the scaffold (so an mbf-unroll group emits
    // all cores first, then the folds in strict ascending order). ----
    auto emitQ40Core =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<FlatBlockCore> {
      // Fresh per-core base memo: each unrolled core (blockOffset k) recomputes
      // base + (ib+k)*stride; a memo shared across cores would alias block k>0 to
      // block 0's address (byte-diff AND a correctness bug).
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
        mlir::Value idx = ib;
        if (blockOffset != 0)
          idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                              sizeLit(blockOffset));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };
      mlir::Value xb = blockBaseFor(
          q40WeightLoad.getBuffer(), q40WeightLoad.getBlockIndex(),
          static_cast<int64_t>(*q40WeightLoad.getBlockStride()), "block_base_x");
      mlir::Value yb = blockBaseFor(
          q40LowLoad.getBuffer(), q40LowLoad.getBlockIndex(),
          static_cast<int64_t>(*q40LowLoad.getBlockStride()), "block_base_y");
      auto fp16ReadAt = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
        mlir::Value addr = blockBase;
        if (byteOffset && *byteOffset != 0)
          addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                               blockBase, sizeLit(*byteOffset));
        return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                              mlir::ValueRange{addr}, opName, role,
                              llvm::StringRef("fcvt.s.h"));
      };
      // item4 (fcvt.s.h reschedule): the two per-block fp16 scale reads (dX, dY)
      // are DEFERRED to after the integer core, right before the fold -- matching
      // the ggml factory placement. Only the fcvt READ moves; block_base_x/y stays
      // here (memoized, shared with the integer-core loads) and the fold
      // arithmetic/rounding order is untouched, so values are byte-identical
      // (fold-oracle §1 preserved). Reads emitted just before the return.
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // One strip: op-sourced loads + offset-binary decode product + reduce +
      // lane0 extract + assign sumi. chunkOffset is the elided 0 literal or the
      // robust strip induction var; carrySumi seeds lane0 from the sumi lvalue
      // (robust re-strip) or a fresh 0 (elided single cover).
      auto emitStrip = [&](mlir::Value chunkOffset, mlir::Value vl,
                           bool carrySumi) -> mlir::LogicalResult {
        auto emitLoadTail = [&](mlir::Value blockBase,
                                int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[q40WeightLoad.getLoaded()] = emitLoadTail(
            xb, static_cast<int64_t>(*q40WeightLoad.getQuantByteOffset()));
        valueMap[q40LowLoad.getLoaded()] = emitLoadTail(
            yb, flatPlan->activationQuantOffset);
        valueMap[q40HighLoad.getLoaded()] = emitLoadTail(
            yb, flatPlan->activationQuantOffset + flatPlan->blockLen);
        mlir::FailureOr<mlir::Value> productOr =
            emitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(packedProduct.getWeight()),
                valueMap.lookup(packedProduct.getActivationLow()),
                valueMap.lookup(packedProduct.getActivationHigh()), vl,
                i8CoreType, i16WideType, "i8", coreLmul, 16, wideLmul, "i16",
                opName, role);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[packedProduct.getResult()] = *productOr;
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
              return {sumiSeed, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, "__riscv_vmv_x_s_i32m1_i32",
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);
        return mlir::success();
      };

      bool robust = forceRobust || !stripElided;
      if (!robust) {
        // Elided single cover: ONE vsetvl(block_len) caps the active vl at the
        // whole half-block, ONE strip reduce, no inner loop, no sumi carry.
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        if (mlir::failed(emitStrip(sizeLit(0), vl, /*carrySumi=*/false)))
          return mlir::failure();
      } else {
        // Robust inner strip loop: step = loop-invariant VLMAX, per-chunk active
        // vl = vsetvl(block_len - c), sumi-carrying seed. VLEN-robust re-strip.
        mlir::Value innerVlmax = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        auto innerLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
            /*bodyBuilder=*/nullptr);
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
        if (mlir::failed(emitStrip(c, vl, /*carrySumi=*/true)))
          return mlir::failure();
      }
      // item4 (fcvt.s.h reschedule): deferred per-block fp16 scale reads. The
      // block_base_x/y arithmetic was emitted (and memoized) above, shared with
      // the integer-core loads, so blockBaseFor here HITS the memo -- these two
      // calls re-emit ONLY the fcvt.s.h reads, now placed AFTER the integer core /
      // right before the fold. Values byte-identical; only fcvt position moves.
      mlir::Value dX = fp16ReadAt(
          blockBaseFor(brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                       "block_base_x"),
          brick1.getLhsScaleByteOffset());
      mlir::Value dY = fp16ReadAt(
          blockBaseFor(brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                       "block_base_y"),
          brick1.getRhsScaleByteOffset());
      return FlatBlockCore{sumiVar.getResult(), dX, dY, mlir::Value(),
                           mlir::Value()};
    };

    // ---- The mbf schedule scaffold (mirrors emitFlatBlockDot): mbf==1 = single
    // block loop {core; fold}; mbf>1 = a main loop stepping by factor emitting
    // ALL factor cores FIRST then the factor folds in STRICT ascending block
    // order (the fp non-associativity boundary), plus a robust single-block
    // scalar tail over the nb % factor remainder. ----
    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<FlatBlockCore> core =
          emitQ40Core(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                   core->mX, core->sY);
    } else {
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nbMain, factorLit, /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<FlatBlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<FlatBlockCore> core =
              emitQ40Core(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const FlatBlockCore &core : cores)
          emitFlatFold(rewriter, loc, st, core.sumiVar, core.dX, core.dY,
                       core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(
          loc, nbMain, nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<FlatBlockCore> core =
            emitQ40Core(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                     core->mX, core->sY);
      }
    }
  } else if (isQ80ScheduleParam) {
    // ===================================================================
    // q8_0 (sumi_times_scales) SCHEDULE-PARAMETERIZED emit: the FULL legal
    // {integer_core_lmul=m2, multi_block_factor, strip_elision} cross product,
    // driven by the loop-body knobs but sourced OP-BY-OP from the region ops
    // (the brick chain drives WHAT computes; the knobs drive HOW it schedules).
    // Ported from the q4_0 schedule scaffold: q8_0's core is the whole-block
    // plain-i8 signed widening product (m2 native, blockLen = qk, two i8 loads,
    // direct vwmul) instead of the half-block packed-i4 decode; the mbf-unroll
    // main loop + strict-ascending folds + robust tail scaffold is structurally
    // identical, so each legal combo is byte-identical to the monolith's
    // same-knob GgmlBlockDotQ80Q80Op (emitFlatBlockDot) instance while the
    // integer core stays region-driven (anti-bypass W4). ==================
    mlir::Block &coreBlock = loopBody.getBody().front();

    // ---- Region walk (identify, no emit) ----
    weftrvv::BlockFp16ScaleProductOp brick1;
    weftrvv::BlockComputedScaleDequantOp brick2 = peekBrick2;
    weftrvv::CrossBlockF32AccumulateOp brick3;
    weftrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    weftrvv::WideningProductOp coreProduct;
    weftrvv::StandaloneReduceOp coreReduce;
    weftrvv::TypedVectorLane0ToScalarExtractOp coreExtract;
    llvm::SmallVector<weftrvv::LoadOp, 2> coreLoads;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::BlockFp16ScaleProductOp>(bodyOp))
        brick1 = o;
      else if (auto o = llvm::dyn_cast<weftrvv::CrossBlockF32AccumulateOp>(bodyOp))
        brick3 = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::WideningProductOp>(bodyOp))
        coreProduct = o;
      else if (auto o = llvm::dyn_cast<weftrvv::StandaloneReduceOp>(bodyOp))
        coreReduce = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedVectorLane0ToScalarExtractOp>(
                       bodyOp))
        coreExtract = o;
      else if (auto o = llvm::dyn_cast<weftrvv::LoadOp>(bodyOp))
        coreLoads.push_back(o);
    });

    // ---- Region-driven gate (fail-closed, I7): the SAME brick + integer-core
    // wiring the mbf==1 else path checks, so the schedule-parameterized emit
    // provably tracks the region content, not the attrs. ----
    if (!brick1 || !brick3 || !yieldOp)
      return rewriter.notifyMatchFailure(
          loopBody, "q8_0 schedule-parameterized body requires brick 1, brick 2, "
                    "brick 3, and the yield");
    if (brick1.getBlockIndex() != coreBlock.getArgument(0))
      return rewriter.notifyMatchFailure(
          brick1, "brick 1 block_index must be the loop induction variable");
    if (brick2.getComputedScale() != brick1.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 computed_scale must be brick 1's per-block scale");
    if (brick3.getAcc() != coreBlock.getArgument(1))
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 acc must be the loop-carried accumulator");
    if (brick3.getTerm() != brick2.getResult())
      return rewriter.notifyMatchFailure(
          brick3, "brick 3 term must be brick 2's per-block dequant term");
    if (yieldOp.getAccNext() != brick3.getResult())
      return rewriter.notifyMatchFailure(
          yieldOp, "loop yield acc_next must be brick 3's fold result");
    if (coreLoads.size() != 2 || !coreProduct || !coreReduce || !coreExtract)
      return rewriter.notifyMatchFailure(
          loopBody,
          "full q8_0 flat block-dot body requires the region integer core: two "
          "per-block i8 loads, a signed widening product, a standalone reduce, "
          "and a lane0 scalar extract");
    if (brick2.getSumi() != coreExtract.getResult())
      return rewriter.notifyMatchFailure(
          brick2, "brick 2 sumi must be the integer-core lane0 extract result");

    // Identify the weight vs activation per-block load by ABI buffer; both must
    // carry the per-block block_stride + quant_byte_offset the address
    // arithmetic depends on (the operand-flow real gate).
    weftrvv::LoadOp weightLoad, activationLoad;
    for (weftrvv::LoadOp ld : coreLoads) {
      if (ld.getBuffer() == loopBody.getWeightBase())
        weightLoad = ld;
      else if (ld.getBuffer() == loopBody.getActivationBase())
        activationLoad = ld;
    }
    if (!weightLoad || !activationLoad || !weightLoad.getQuantByteOffset() ||
        !activationLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
        !activationLoad.getBlockStride() ||
        !matchesFlatPlanSingleActivationOffset(activationLoad))
      return rewriter.notifyMatchFailure(
          loopBody, "q8_0 integer-core loads must read the weight and activation "
                    "ABI buffers off the loop body, each carrying a block_stride "
                    "+ quant_byte_offset");
    if (coreProduct.getKind() != "signed_widening_product")
      return rewriter.notifyMatchFailure(
          coreProduct, "q8_0 integer-core widening product must be a signed "
                       "widening product");
    mlir::Value pl = coreProduct.getLhs(), pr = coreProduct.getRhs();
    mlir::Value wLoaded = weightLoad.getLoaded(),
                aLoaded = activationLoad.getLoaded();
    if (!((pl == wLoaded && pr == aLoaded) ||
          (pl == aLoaded && pr == wLoaded)))
      return rewriter.notifyMatchFailure(
          coreProduct, "q8_0 integer-core widening product operands must be the "
                       "two per-block load results");
    if (coreReduce.getInput() != coreProduct.getResult())
      return rewriter.notifyMatchFailure(
          coreReduce,
          "q8_0 integer-core reduce input must be the widening product");
    if (coreExtract.getInput() != coreReduce.getResult())
      return rewriter.notifyMatchFailure(
          coreExtract, "q8_0 integer-core lane0 extract input must be the "
                       "reduce");

    // ---- The pinned-oracle SeparatedLeftAssoc fold descriptor + shared emit
    // state (fold only; the integer core is emitted op-by-op below). The q8_0
    // TYPED body conforms to [measurement/浮点折叠oracle.md §1]:
    // ((sumi*d_x)*d_y) as SEPARATE emitc statements (no dx*dy premultiply, no FMA
    // contraction). The MONOLITH keeps SumiTimesScales (`sumi*(d_x*d_y)`), so the
    // typed emit is INTENTIONALLY no longer byte-exact vs monolith here -- the
    // sanctioned gate migration (monolith retires later), q8_0 only. ----
    FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
    BlockDotFacts facts = *finalSchedule;
    FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
        rewriter, descriptor, facts, weightBase, activationBase,
        sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType, opName,
        role);

    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    llvm::StringRef coreLmul =
        llvm::cast<weftrvv::VectorType>(weightLoad.getLoaded().getType())
            .getLmul();
    llvm::StringRef wideLmul =
        llvm::cast<weftrvv::VectorType>(coreProduct.getResult().getType())
            .getLmul();
    mlir::Type i8CoreType =
        emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type i16WideType =
        emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
    std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
    unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
    llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
    std::string innerSetvlCallee =
        riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
    int64_t blockLen = flatPlan->blockLen;

    // ---- The per-block q8_0 core, parameterized by (ib, blockOffset,
    // forceRobust). Mirrors emitQ40Core: emits address / brick-1 scales / the
    // op-by-op integer core (elided single cover OR robust inner strip loop) and
    // returns the fold inputs; the FOLD is emitted by the scaffold (so an
    // mbf-unroll group emits all cores first, then the folds in strict ascending
    // order). ----
    auto emitQ80Core =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<FlatBlockCore> {
      // Fresh per-core base memo: each unrolled core (blockOffset k) recomputes
      // base + (ib+k)*stride; a memo shared across cores would alias block k>0 to
      // block 0's address.
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
        mlir::Value idx = ib;
        if (blockOffset != 0)
          idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                              sizeLit(blockOffset));
        mlir::Value off =
            rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
        mlir::Value base = rewriter.create<emitc::AddOp>(
            loc, emittedBase.getType(), emittedBase, off);
        blockBaseMemo[key] = base;
        return base;
      };
      mlir::Value xb = blockBaseFor(
          weightLoad.getBuffer(), weightLoad.getBlockIndex(),
          static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
      mlir::Value yb = blockBaseFor(
          activationLoad.getBuffer(), activationLoad.getBlockIndex(),
          static_cast<int64_t>(*activationLoad.getBlockStride()), "block_base_y");
      auto fp16ReadAt = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
        mlir::Value addr = blockBase;
        if (byteOffset && *byteOffset != 0)
          addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                               blockBase, sizeLit(*byteOffset));
        return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                              mlir::ValueRange{addr}, opName, role,
                              llvm::StringRef("fcvt.s.h"));
      };
      // item4 (fcvt.s.h reschedule): the two per-block fp16 scale reads (dX, dY)
      // are DEFERRED to after the integer core, right before the fold -- matching
      // the ggml factory placement (the scalar fp16->f32 conversion sits at the
      // fold point, not early in the integer-core vector section). Only the fcvt
      // READ moves: block_base_x/block_base_y arithmetic stays here (memoized and
      // shared with the integer-core loads), and the fold arithmetic/rounding
      // order is untouched, so the emitted VALUES are byte-identical (fold-oracle
      // §1 preserved). The deferred reads are emitted just before the return.
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // One strip: op-sourced weight+activation i8 loads + plain-i8 signed
      // widening vwmul product + reduce + lane0 extract + assign sumi.
      // chunkOffset is the elided 0 literal or the robust strip induction var;
      // carrySumi seeds lane0 from the sumi lvalue (robust re-strip) or a fresh 0
      // (elided single cover).
      auto emitStrip = [&](mlir::Value chunkOffset, mlir::Value vl,
                           bool carrySumi) -> mlir::LogicalResult {
        auto emitLoadTail = [&](mlir::Value blockBase,
                                int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTail(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[activationLoad.getLoaded()] = emitLoadTail(
            yb, flatPlan->activationQuantOffset);
        // The signed widening vwmul, emitted from ITS lhs/rhs load-result
        // operands (via the valueMap) -- byte-exact to the monolith's
        // `__riscv_vwmul_vv_i16m4(vx0, vy0, vl)` (lhs=weight first).
        valueMap[coreProduct.getResult()] = emitOpaqueCall(
            rewriter, loc, i16WideType, mulCallee,
            mlir::ValueRange{valueMap.lookup(coreProduct.getLhs()),
                             valueMap.lookup(coreProduct.getRhs()), vl},
            opName, role);
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
              return {sumiSeed, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, "__riscv_vmv_x_s_i32m1_i32",
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);
        return mlir::success();
      };

      bool robust = forceRobust || !stripElided;
      if (!robust) {
        // Elided single cover: ONE vsetvl(block_len) caps the active vl at the
        // whole block, ONE strip reduce, no inner loop, no sumi carry.
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        if (mlir::failed(emitStrip(sizeLit(0), vl, /*carrySumi=*/false)))
          return mlir::failure();
      } else {
        // Robust inner strip loop: step = loop-invariant VLMAX, per-chunk active
        // vl = vsetvl(block_len - c), sumi-carrying seed. VLEN-robust re-strip.
        mlir::Value innerVlmax = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        auto innerLoop = rewriter.create<emitc::ForOp>(
            loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
            /*bodyBuilder=*/nullptr);
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
        if (mlir::failed(emitStrip(c, vl, /*carrySumi=*/true)))
          return mlir::failure();
      }
      // item4 (fcvt.s.h reschedule): deferred per-block fp16 scale reads. The
      // block_base_x/y arithmetic was already emitted (and memoized) above, shared
      // with the integer-core loads, so blockBaseFor here HITS the memo -- these
      // two calls re-emit ONLY the (float)*(const _Float16 *) fcvt.s.h reads, now
      // placed AFTER the integer core / right before the fold (ggml factory
      // placement). Values are byte-identical; only the fcvt code position moves.
      mlir::Value dX = fp16ReadAt(
          blockBaseFor(brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                       "block_base_x"),
          brick1.getLhsScaleByteOffset());
      mlir::Value dY = fp16ReadAt(
          blockBaseFor(brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                       static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                       "block_base_y"),
          brick1.getRhsScaleByteOffset());
      return FlatBlockCore{sumiVar.getResult(), dX, dY, mlir::Value(),
                           mlir::Value()};
    };

    // ---- M-FLAT P2c: the deferred-ordered fold_structure. SAME pinned §1 oracle
    // (measurement/浮点折叠oracle.md), issued as a batched VECTOR
    // reduction instead of the per-block scalar fold. For a batch of B =
    // multi_block_factor blocks: PHASE A (out-of-order free) runs the B per-block
    // integer cores (region-sourced vwmul -> vwredsum -> lane0 extract -> scalar
    // sumi_k) and packs the B sumi into ONE i32m1 vector via vslide1down (vl=B:
    // each slide inserts sumi_k at lane B-1 shifting prior lanes down, so the
    // final lane i == block i in STRICT ASCENDING order), plus a vlse16 strided
    // load of the B fp16 d_x / d_y scales (block stride) widened f16->f32 (exact).
    // PHASE B (bit-exact by construction) converts sumi (int32->f32, exact since
    // |sumi| <= 32*127*128 < 2^24), does vfmul x2 per lane (t_b = ((sumi_b*d_x_b)
    // *d_y_b), the SeparatedLeftAssoc §1 tree -- NO d_x*d_y premultiply, NO FMA
    // contraction: vfmul/vfredosum are SEPARATE ops), then folds the B terms with
    // ONE vfredosum.vs SEEDED by the running sumf. RVV vfredosum.vs reduces
    // lane-ascending, seed-first: ((sumf + t_0) + t_1) + ... + t_{B-1} = the §1
    // serial left-fold BYTE-FOR-BYTE. Batch chaining (each batch seeds vfredosum
    // with the prior sumf) + the nb % B robust scalar tail (the SAME §1
    // SeparatedLeftAssoc scalar fold) reconstruct the full §1 fold. All PHASE-B
    // ops use vl=B so inactive lanes never contribute. Deferred requires
    // multi_block_factor in {2,4} and the elided single-cover integer core
    // (VLEN>=128 covers qk=32 in one i8m2 strip); fail-closed otherwise. The perf
    // mechanism (the per-block vwredsum -> vmv.x.s -> scalar-fmaf cross-domain
    // round-trip disappears into one vfredosum, so batch k's ordered fold can
    // overlap batch k+1's integer phase) is HYPOTHESIZED / pending-hardware; this
    // step establishes constructibility + §1 bit-exactness + lit only. The
    // per-block integer vwredsum is UNCHANGED -- only the cross-block FP fold is
    // vectorized. ----
    if (foldStructure == "deferred-ordered") {
      if (multiBlockFactor != 2 && multiBlockFactor != 4)
        return rewriter.notifyMatchFailure(
            loopBody,
            "deferred-ordered fold_structure requires multi_block_factor 2 or 4 "
            "(the vector batch size B); mbf==1 is the degenerate single-lane "
            "reduction, not materialized");
      if (!stripElided)
        return rewriter.notifyMatchFailure(
            loopBody,
            "deferred-ordered fold_structure requires strip_elision \"elided\" "
            "(the whole-block i8m2 single-cover integer core; VLEN>=128 covers qk "
            "in one strip) -- a robust inner strip would disagree with the elided "
            "emit (attribute-derived-emission lie)");

      const int64_t B = multiBlockFactor;
      mlir::Type f32m1Type = emitc::OpaqueType::get(ctx, "vfloat32m1_t");
      mlir::Type f16mf2Type = emitc::OpaqueType::get(ctx, "vfloat16mf2_t");
      mlir::Type f16PtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const _Float16"));
      std::string sumiSeedCallee =
          riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      std::string slideCallee =
          riscvScalarImmediateIntrinsicName("vslide1down_vx", "i32", "m1");
      std::string vlse16Callee = riscvIntrinsicName("vlse", 16, "mf2", "f16");
      std::string scaleWidenCallee =
          riscvFloatWideningConvertIntrinsicName("f32", "m1");
      std::string sumiCvtCallee =
          riscvIntrinsicName("vfcvt_f_x_v", 32, "m1", "f32");
      std::string fmulCallee = riscvIntrinsicName("vfmul", 32, "m1", "f32");
      std::string fredosumCallee =
          riscvReductionIntrinsicName("vfredosum", 32, "m1", "f32");
      std::string fSeedCallee = riscvIntrinsicName("vfmv_v_f", 32, "m1", "f32");
      std::string fExtractCallee =
          riscvFloatScalarExtractIntrinsicName("f32", "m1");
      // [GAP-NUM] relaxed-tier callees (§5 reassociation variant): a fused
      // vfmacc.vv (one rounding for sumi*dxy + accumulate) into a persistent
      // lane-wise accumulator, collapsed by ONE unordered tree vfredusum.vs at the
      // very end. Both are BANNED in the strict tier (the strict deferred lit pins
      // implicit-check-not vfmacc/vfredusum); they materialize only when relaxed.
      const bool relaxed = numericsTier == "relaxed";
      std::string vfmaccCallee = riscvIntrinsicName("vfmacc", 32, "m1", "f32");
      std::string fredusumCallee =
          riscvReductionIntrinsicName("vfredusum", 32, "m1", "f32");

      // Region-sourced strides / offsets (the operand-flow real gate, W4): the
      // per-block integer-core address arithmetic reads the load ops' strides,
      // the scale strided loads read brick 1's scale strides.
      int64_t wStride = static_cast<int64_t>(*weightLoad.getBlockStride());
      int64_t aStride = static_cast<int64_t>(*activationLoad.getBlockStride());
      int64_t wQuantOff = static_cast<int64_t>(*weightLoad.getQuantByteOffset());
      int64_t aQuantOff = flatPlan->activationQuantOffset;
      int64_t dxScaleStride =
          static_cast<int64_t>(brick1.getLhsBlockStride().value_or(wStride));
      int64_t dyScaleStride =
          static_cast<int64_t>(brick1.getRhsBlockStride().value_or(aStride));
      int64_t dxScaleOff =
          static_cast<int64_t>(brick1.getLhsScaleByteOffset().value_or(0));
      int64_t dyScaleOff =
          static_cast<int64_t>(brick1.getRhsScaleByteOffset().value_or(0));
      mlir::Value dxScaleBase = valueMap.lookup(brick1.getLhsScaleBase());
      mlir::Value dyScaleBase = valueMap.lookup(brick1.getRhsScaleBase());
      bool weightIsLhs = coreProduct.getLhs() == weightLoad.getLoaded();

      // One block's elided single-cover integer core (region-sourced): address ->
      // two i8m2 loads -> vwmul -> vwredsum(seed 0) -> lane0 extract -> scalar
      // sumi. Byte-identical inner ops to emitQ80Core's elided strip; returns the
      // scalar sumi_k (NOT folded -- the deferred phase packs it into the vector).
      auto emitDeferredBlockSumi = [&](mlir::Value blockIdx) -> mlir::Value {
        auto blockBaseFor = [&](mlir::Value base, int64_t stride,
                                const char *step) -> mlir::Value {
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off = rewriter.create<emitc::MulOp>(loc, sizeType, blockIdx,
                                                          sizeLit(stride));
          return rewriter.create<emitc::AddOp>(loc, base.getType(), base, off);
        };
        mlir::Value xb = blockBaseFor(weightBase, wStride, "block_base_x");
        mlir::Value yb = blockBaseFor(activationBase, aStride, "block_base_y");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(blockLen)};
            });
        auto loadI8 = [&](mlir::Value blockBase,
                          int64_t quantOff) -> mlir::Value {
          mlir::Value addr = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, addr).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        mlir::Value wv = loadI8(xb, wQuantOff);
        mlir::Value av = loadI8(yb, aQuantOff);
        mlir::Value prod = emitOpaqueCall(
            rewriter, loc, i16WideType, mulCallee,
            mlir::ValueRange{weightIsLhs ? wv : av, weightIsLhs ? av : wv, vl},
            opName, role);
        mlir::Value zeroI32 =
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
        mlir::Value seed =
            emitOpaqueCall(rewriter, loc, i32m1Type, sumiSeedCallee,
                           mlir::ValueRange{zeroI32, sizeLit(1)}, opName, role);
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red =
            emitOpaqueCall(rewriter, loc, i32m1Type, reduceCallee,
                           mlir::ValueRange{prod, seed, vl}, opName, role);
        return emitOpaqueCall(rewriter, loc, i32Type,
                              "__riscv_vmv_x_s_i32m1_i32",
                              mlir::ValueRange{red}, opName, role);
      };

      // [GAP-NUM] relaxed tier: a PERSISTENT B-lane fp32 accumulator held across
      // ALL batches (the K-way accumulator of §5). Seeded to 0.0 once before the
      // main loop; each batch fuses its B lane terms in via vfmacc; the whole
      // vector collapses to sumf with ONE unordered vfredusum AFTER the loop. The
      // strict tier keeps sumf as its only accumulator (this lvalue is unused).
      mlir::Value accVecVar;
      if (relaxed) {
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("acc_vec", opName, role));
        accVecVar = rewriter
                        .create<emitc::VariableOp>(
                            loc, emitc::LValueType::get(f32m1Type),
                            emitc::OpaqueAttr::get(ctx, ""))
                        .getResult();
        mlir::Value zeroF =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
        mlir::Value zeroVec =
            emitOpaqueCall(rewriter, loc, f32m1Type, fSeedCallee,
                           mlir::ValueRange{zeroF, sizeLit(B)}, opName, role);
        rewriter.create<emitc::AssignOp>(loc, accVecVar, zeroVec);
      }

      // The by-B main loop: each batch packs B sumi + strided B scales, then ONE
      // seed-ordered vfredosum fold; the nb % B robust scalar tail continues the
      // SAME §1 serial fold.
      mlir::Value factorLit = sizeLit(B);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nbMain, factorLit, /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        mlir::Value ibBatch = mainLoop.getInductionVar();

        // PHASE A: B per-block integer cores -> vslide1down pack into one i32m1
        // (vl=B: final lane i == block i, ascending). sumi_vec seeded 0.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "deferred_sumi_pack"));
        mlir::Value packZero =
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
        mlir::Value sumiVec =
            emitOpaqueCall(rewriter, loc, i32m1Type, sumiSeedCallee,
                           mlir::ValueRange{packZero, sizeLit(B)}, opName, role);
        for (int64_t k = 0; k < B; ++k) {
          mlir::Value blockIdx = ibBatch;
          if (k != 0)
            blockIdx = rewriter.create<emitc::AddOp>(loc, sizeType, ibBatch,
                                                     sizeLit(k));
          mlir::Value sumiK = emitDeferredBlockSumi(blockIdx);
          sumiVec = emitOpaqueCall(
              rewriter, loc, i32m1Type, slideCallee,
              mlir::ValueRange{sumiVec, sumiK, sizeLit(B)}, opName, role);
        }

        // PHASE A scales: vlse16 the B d_x / d_y fp16 scales (block stride),
        // widen f16->f32 (exact). Lane order == the sumi_vec lane order (both
        // batch-base + k*stride ascending).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "deferred_scale_gather"));
        auto scaleVec = [&](mlir::Value base, int64_t stride,
                            int64_t off) -> mlir::Value {
          mlir::Value batchOff = rewriter.create<emitc::MulOp>(
              loc, sizeType, ibBatch, sizeLit(stride));
          mlir::Value addr =
              rewriter.create<emitc::AddOp>(loc, base.getType(), base, batchOff);
          if (off != 0)
            addr = rewriter.create<emitc::AddOp>(loc, base.getType(), addr,
                                                 sizeLit(off));
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, f16PtrType, addr).getResult();
          mlir::Value f16v = emitOpaqueCall(
              rewriter, loc, f16mf2Type, vlse16Callee,
              mlir::ValueRange{ptr, sizeLit(stride), sizeLit(B)}, opName, role);
          return emitOpaqueCall(rewriter, loc, f32m1Type, scaleWidenCallee,
                                mlir::ValueRange{f16v, sizeLit(B)}, opName, role);
        };
        mlir::Value dxVec = scaleVec(dxScaleBase, dxScaleStride, dxScaleOff);
        mlir::Value dyVec = scaleVec(dyScaleBase, dyScaleStride, dyScaleOff);

        if (relaxed) {
          // PHASE B (RELAXED, §5). The reassociation variant: premultiply the two
          // scales into d_xy (ONE vfmul -- §1 FORBIDS d_x*d_y premultiply, §5
          // ALLOWS it) and fuse sumi*d_xy into the persistent lane-wise
          // accumulator with ONE vfmacc (a single fused rounding replacing the §1
          // {vfmul, vfmul} + the per-batch ordered reduction). NO per-batch
          // reduction -- the whole cross-block collapse is deferred to one
          // post-loop vfredusum, so the serial fold chain is broken (batches are
          // independent). Verified against the reassoc-tolerant oracle + a declared
          // ULP bound, NEVER §1.
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "relaxed_reassoc_fold"));
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m1Type, sumiCvtCallee,
                             mlir::ValueRange{sumiVec, sizeLit(B)}, opName, role);
          mlir::Value dxyVec = emitOpaqueCall(
              rewriter, loc, f32m1Type, fmulCallee,
              mlir::ValueRange{dxVec, dyVec, sizeLit(B)}, opName, role);
          mlir::Value accCur =
              rewriter.create<emitc::LoadOp>(loc, f32m1Type, accVecVar)
                  .getResult();
          mlir::Value accNext = emitOpaqueCall(
              rewriter, loc, f32m1Type, vfmaccCallee,
              mlir::ValueRange{accCur, sumiF, dxyVec, sizeLit(B)}, opName, role);
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("acc_vec", opName, role));
          rewriter.create<emitc::AssignOp>(loc, accVecVar, accNext);
        } else {
          // PHASE B (STRICT): bit-exact §1. int32->f32 (exact), vfmul x2 per lane
          // (SEPARATE roundings -- NO premultiply, NO FMA), then ONE vfredosum.vs
          // SEEDED by the running sumf (lane-ascending, seed-first = serial
          // left-fold). This block is BYTE-IDENTICAL to the pre-[GAP-NUM] strict
          // deferred emit (op order unchanged).
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "deferred_ordered_fold"));
          mlir::Value sumfCur =
              rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
          mlir::Value seedVec =
              emitOpaqueCall(rewriter, loc, f32m1Type, fSeedCallee,
                             mlir::ValueRange{sumfCur, sizeLit(1)}, opName, role);
          mlir::Value sumiF =
              emitOpaqueCall(rewriter, loc, f32m1Type, sumiCvtCallee,
                             mlir::ValueRange{sumiVec, sizeLit(B)}, opName, role);
          mlir::Value tVec = emitOpaqueCall(
              rewriter, loc, f32m1Type, fmulCallee,
              mlir::ValueRange{sumiF, dxVec, sizeLit(B)}, opName, role);
          tVec = emitOpaqueCall(rewriter, loc, f32m1Type, fmulCallee,
                                mlir::ValueRange{tVec, dyVec, sizeLit(B)}, opName,
                                role);
          mlir::Value red = emitOpaqueCall(
              rewriter, loc, f32m1Type, fredosumCallee,
              mlir::ValueRange{tVec, seedVec, sizeLit(B)}, opName, role);
          mlir::Value sumfNext =
              emitOpaqueCall(rewriter, loc, floatType, fExtractCallee,
                             mlir::ValueRange{red}, opName, role);
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumf", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfNext);
        }
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(
          loc, nbMain, nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<FlatBlockCore> core =
            emitQ80Core(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                     core->mX, core->sY);
      }
      if (relaxed) {
        // [GAP-NUM] relaxed final collapse: ONE unordered vfredusum.vs folds the B
        // deferred lane accumulators into the running sumf (which by now already
        // carries the nb % B strict scalar tail). vfredusum is a TREE reduction --
        // it does NOT preserve lane order, which is exactly the reassociation §5
        // permits (and §1 forbids). Seeded by sumf so the vector part folds onto
        // the scalar tail; the result lands back in the SAME sumf lvalue the store
        // reads.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "relaxed_final_reduce"));
        mlir::Value sumfCur =
            rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
        mlir::Value seedVec =
            emitOpaqueCall(rewriter, loc, f32m1Type, fSeedCallee,
                           mlir::ValueRange{sumfCur, sizeLit(1)}, opName, role);
        mlir::Value accCur =
            rewriter.create<emitc::LoadOp>(loc, f32m1Type, accVecVar).getResult();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, f32m1Type, fredusumCallee,
            mlir::ValueRange{accCur, seedVec, sizeLit(B)}, opName, role);
        mlir::Value sumfNext =
            emitOpaqueCall(rewriter, loc, floatType, fExtractCallee,
                           mlir::ValueRange{red}, opName, role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumf", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfNext);
      }
    } else if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<FlatBlockCore> core =
          emitQ80Core(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                   core->mX, core->sY);
    } else {
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), nbMain, factorLit, /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<FlatBlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<FlatBlockCore> core =
              emitQ80Core(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const FlatBlockCore &core : cores)
          emitFlatFold(rewriter, loc, st, core.sumiVar, core.dX, core.dY,
                       core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(
          loc, nbMain, nb, sizeLit(1), /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<FlatBlockCore> core =
            emitQ80Core(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                     core->mX, core->sY);
      }
    }
  } else {
    // ---- The remaining folds (q4_1 / q5_0 / q5_1) currently materialize only
    // the mbf==1 + elided default; q8_0 (sumi_times_scales) and q4_0
    // (left_assoc) take the schedule-parameterized branches above. ----
    if (multiBlockFactor != 1 || !stripElided)
      return rewriter.notifyMatchFailure(
          loopBody,
          "multi_block_factor>1 and the robust strip form are currently "
          "materialized only for the left_assoc (q4_0) and sumi_times_scales "
          "(q8_0) flat bodies; the other folds require the "
          "multi_block_factor==1 + strip_elision==elided is the materialized "
          "plan for this fold family");

  // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- byte-exact to
  // emitFlatBlockDot:5879-5883 (the no-unroll block loop).
  auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb, sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());

    mlir::Block &coreBlock = loopBody.getBody().front();

    // M-FLAT step 5a + W4: recognize the COMPLETE q8_0 typed body in the region
    // and drive the byte-exact FULL-body emit OP-BY-OP from the region ops'
    // operands (NOT套壳 re-derived from the loop-body attrs through the
    // monolithic emitFlatBlockCore). The region carries the FULL per-block chain:
    // brick 1 (the per-block d_x*d_y scale) -> the vector integer core (two
    // per-block i8 loads -> signed widening product -> standalone reduce -> lane0
    // scalar extract into the i32 sumi) -> brick 2 (the computed-scale i32-sumi
    // dequant on that sumi) -> brick 3 (the cross-block fp32 fold) -> yield. The
    // chain is recognized REGION-DRIVEN: the presence of brick 2 marks a full
    // body, and the emit is GATED on the region's actual op WIRING -- a missing
    // brick 2, a fold folded on a non-brick-1 scale, OR a broken/misdirected
    // integer-core link (W4) fails closed here. After the gate, each op is emitted
    // from ITS operands: the load's block_stride/quant_byte_offset drive the
    // address arithmetic, the product reads its lhs/rhs load results, the reduce
    // reads its product, the extract reads its reduce, and brick 1's two fp16
    // reads share the load's per-block base (memoized on the (buffer, block_index)
    // SSA pair). Only the fused fold tail (brick 1's d_x*d_y + brick 2's cast/mul
    // + brick 3's add) collapses COLLECTIVELY into one emitc.expression via the
    // reused emitFlatFold, fed the operand-derived d_x/d_y + sumi. The result is
    // byte-identical to the monolithic q8_0 (verified by golden diff) yet sourced
    // from the region ops, not the attrs.
    weftrvv::BlockFp16ScaleProductOp brick1;
    weftrvv::BlockComputedScaleDequantOp brick2;
    weftrvv::CrossBlockF32AccumulateOp brick3;
    weftrvv::TypedFlatBlockDotLoopYieldOp yieldOp;
    // W4: the region's vector integer-core chain (two per-block i8 loads ->
    // signed widening product -> standalone reduce -> lane0 scalar extract ->
    // the scalar i32 sumi feeding brick 2). Collected so the full-body gate can
    // check every link against the actual region SSA wiring.
    llvm::SmallVector<weftrvv::LoadOp, 2> coreLoads;
    weftrvv::WideningProductOp coreProduct;
    weftrvv::StandaloneReduceOp coreReduce;
    weftrvv::TypedVectorLane0ToScalarExtractOp coreExtract;
    loopBody.getBody().walk([&](mlir::Operation *bodyOp) {
      if (auto o = llvm::dyn_cast<weftrvv::BlockFp16ScaleProductOp>(bodyOp))
        brick1 = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::BlockComputedScaleDequantOp>(bodyOp))
        brick2 = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::CrossBlockF32AccumulateOp>(bodyOp))
        brick3 = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopYieldOp>(bodyOp))
        yieldOp = o;
      else if (auto o = llvm::dyn_cast<weftrvv::LoadOp>(bodyOp))
        coreLoads.push_back(o);
      else if (auto o = llvm::dyn_cast<weftrvv::WideningProductOp>(bodyOp))
        coreProduct = o;
      else if (auto o = llvm::dyn_cast<weftrvv::StandaloneReduceOp>(bodyOp))
        coreReduce = o;
      else if (auto o =
                   llvm::dyn_cast<weftrvv::TypedVectorLane0ToScalarExtractOp>(
                       bodyOp))
        coreExtract = o;
    });

    if (!brick2)
      return rewriter.notifyMatchFailure(
          loopBody, "flat block-dot emission requires a complete typed "
                    "mechanism body; skeleton-only bodies are not a production "
                    "fallback");
    {
      // ---- Full q8_0 body: the REGION-DRIVEN gate (defends against a
      // cosmetic attribute-rederive). Every link is checked against the actual
      // region SSA wiring; any break fails closed so the emit provably tracks
      // the region content, not just the loop-body attrs. ----
      if (!brick1 || !brick3 || !yieldOp)
        return rewriter.notifyMatchFailure(
            loopBody, "full flat block-dot body requires brick 1 (per-block "
                      "scale), brick 2 (computed-scale dequant), brick 3 "
                      "(cross-block fold), and the yield");
      if (brick1.getBlockIndex() != coreBlock.getArgument(0))
        return rewriter.notifyMatchFailure(
            brick1, "brick 1 block_index must be the loop induction variable");
      if (brick2.getComputedScale() != brick1.getResult())
        return rewriter.notifyMatchFailure(
            brick2, "brick 2 computed_scale must be brick 1's per-block scale");
      if (brick3.getAcc() != coreBlock.getArgument(1))
        return rewriter.notifyMatchFailure(
            brick3, "brick 3 acc must be the loop-carried accumulator");
      if (brick3.getTerm() != brick2.getResult())
        return rewriter.notifyMatchFailure(
            brick3, "brick 3 term must be brick 2's per-block dequant term");
      if (yieldOp.getAccNext() != brick3.getResult())
        return rewriter.notifyMatchFailure(
            yieldOp, "loop yield acc_next must be brick 3's fold result");

      // fold_model COLLISION tiebreaker (M-FLAT q5_1 = cohort LAST cell): q4_1 AND
      // q5_1 BOTH stamp fold_model="scale_plus_min", so fold_model is no longer a
      // unique key. q5_1 = q5_0's five-bit integer core (five-bit product + qh
      // brick, offset-bias OFF) UNIONed with q4_1's MIN term. Detect by op
      // IDENTITY -- a five-bit product + qh brick under the scale_plus_min fold =>
      // q5_1; else the fold_model chain below dispatches q8_0/q4_0/q4_1/q5_0
      // byte-unchanged (q5_0's scales_times_sumi + q4_1's min-only scale_plus_min
      // both fail this predicate).
      weftrvv::FiveBitOffsetBinaryXI8ProductOp q51FiveBitProduct;
      loopBody.getBody().walk([&](weftrvv::FiveBitOffsetBinaryXI8ProductOp o) {
        q51FiveBitProduct = o;
      });
      weftrvv::BlockFiveBitQhSourceOp q51QhBrick;
      loopBody.getBody().walk(
          [&](weftrvv::BlockFiveBitQhSourceOp o) { q51QhBrick = o; });
      weftrvv::BlockFp16MinProductOp q51MinBrick;
      loopBody.getBody().walk(
          [&](weftrvv::BlockFp16MinProductOp o) { q51MinBrick = o; });
      const bool isQ51Body =
          flatPlan->decodePrimitive == FlatDecodePrimitive::FiveBitOffsetBinary &&
          flatPlan->foldModel == FlatFoldModel::ScalePlusMin &&
          !flatPlan->applyOffsetBias && q51FiveBitProduct && q51QhBrick;

      // Fold-tree dispatch. Both branches emit INTO the block loop body and FALL
      // THROUGH to the shared post-loop `*s = sumf` store below (do NOT return
      // here -- a return would skip the store and ship a kernel that computes but
      // never writes). q4_0's left_assoc fold DESCRIPTOR-DRIVES the SHARED
      // emitFlatBlockCore + emitFlatFold (byte-identical to the monolithic q4_0
      // mbf1 body); q8_0's sumi_times_scales takes the op-by-op W4 emit. Any other
      // fold tree fails closed (I7).
      if (isQ51Body) {
        // ---- q5_1 (M-FLAT cohort LAST cell): the UNION of q5_0's five-bit
        // offset-binary integer core (five-bit product + qh 5th-bit brick) and
        // q4_1's Family-B MIN correction, folded through the ScalePlusMin tree.
        // The SINGLE arithmetic delta vs q5_0 is applyOffsetBias=FALSE (the `-16`
        // bias lives in the per-block MIN scale, so NO `vsub 16` is emitted). The
        // MIN reads (m_x/s_y) sit in the q4_1 slot RIGHT AFTER dX/dY and BEFORE
        // the qh halves -- byte-identical to the monolithic q5_1
        // (FiveBitOffsetBinary / half-block / ScalePlusMin, applyOffsetBias=false,
        // m1, elided, mbf 1) emitFlatBlockCore read order
        // (dX,dY,mX,sY,qhLow16,qhHigh16). The decode + fold + qh source + MIN are
        // the mechanisms that formula construction used to produce the final
        // flat_* plan; this branch consumes that plan and gates the same ops. ----
        weftrvv::FiveBitOffsetBinaryXI8ProductOp fiveBitProduct = q51FiveBitProduct;
        weftrvv::BlockFiveBitQhSourceOp qhBrick = q51QhBrick;
        weftrvv::BlockFp16MinProductOp minBrick = q51MinBrick;
        if (!fiveBitProduct || !qhBrick || !minBrick || coreLoads.size() != 3 ||
            !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q5_1 flat block-dot body requires the region integer core: "
              "three per-block loads (a u8 packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric five-bit offset-binary packed-i4 x i8 "
              "product, a lane0 scalar extract, the per-block qh-source brick, "
              "and the per-block MIN brick");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");
        // The KEY Family-B link (I7): brick 2's min_term must be the MIN brick's
        // per-block m_x*s_y product. q5_1's ScalePlusMin fold REQUIRES it (INVERTS
        // q5_0's "must NOT carry a min_term").
        if (brick2.getMinTerm() != minBrick.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "q5_1 brick 2 min_term must be the per-block MIN brick's "
                      "m_x*s_y correction product");
        // The KEY five-bit link (I7): the product's qh_source operand must be the
        // per-block qh brick's gate-only token.
        if (fiveBitProduct.getQhSource() != qhBrick.getResult())
          return rewriter.notifyMatchFailure(
              fiveBitProduct, "q5_1 five-bit product qh_source must be the "
                              "per-block qh-source brick's gate-only token");

        // Follow the five-bit product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight (u8), q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh.
        auto weightLoad =
            fiveBitProduct.getWeight().getDefiningOp<weftrvv::LoadOp>();
        auto lowLoad =
            fiveBitProduct.getActivationLow().getDefiningOp<weftrvv::LoadOp>();
        auto highLoad =
            fiveBitProduct.getActivationHigh().getDefiningOp<weftrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride() ||
            !matchesFlatPlanHalfActivationOffsets(lowLoad, highLoad))
          return rewriter.notifyMatchFailure(
              loopBody,
              "q5_1 five-bit product operands must be the region's per-block "
              "loads: a u8 packed-i4 weight load off the weight ABI buffer + two "
              "plain-i8 q8 (low/high) activation loads off the activation ABI "
              "buffer, each carrying a block_stride + quant_byte_offset");

        if (!coreReduce ||
            coreReduce.getInput() != fiveBitProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q5_1 integer-core reduce input must be the five-bit product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q5_1 integer-core lane0 extract input must be the "
                           "reduce");

        // The ScalePlusMin fold descriptor with applyOffsetBias=FALSE (the ONE
        // arithmetic delta vs q5_0). Only descriptor.foldModel + the shared state
        // are consumed by emitFlatFold; the integer core + the qh + MIN harvest
        // are emitted op-by-op below.
        FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
        BlockDotFacts facts = *finalSchedule;
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // ONE shared per-block base memo across the merged sub-paths (brick 1's +
        // the MIN brick's + the qh brick's reads and the per-block loads all name
        // the same (%buffer, %block_index) SSA pair -> the memo must hit once; a
        // double-declared base re-emits the `base + ib*stride` and byte-diffs).
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1's two per-block fp16 -> f32 SCALE reads (d_x/d_y at offset 0).
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        // item4 (fcvt.s.h reschedule): brick 1's d_x/d_y and the MIN brick's
        // m_x/s_y fp16->f32 reads are DEFERRED to after the integer core, right
        // before the fold (ggml factory placement). Only the fcvt.s.h reads move;
        // block_base_x/y (memoized, shared with the loads + the qh reads) and the
        // fold arithmetic/rounding stay put, so the emitted VALUES are
        // byte-identical (fold-oracle §1 preserved). The deferred reads are emitted
        // just before emitFlatFold below.

        // The qh field's TWO aligned 16-bit halves, read AFTER m_x/s_y and BEFORE
        // the sumi decl. Off the SHARED weight base (memo hit) at the qh brick's
        // OWN qh_byte_offset -- anti-bypass on the qh brick's qh_base/qh_byte_offset;
        // the descriptor qh offset is NEVER read here. NOT fp16ReadAt: the qh read
        // is a raw `(uint16_t)*(const uint16_t *)` call to a u32.
        mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
        llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
        mlir::Value qhBase = blockBaseFor(
            qhBrick.getQhBase(), qhBrick.getBlockIndex(),
            static_cast<int64_t>(qhBrick.getBlockStride().value_or(0)),
            "block_base_x");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_field"));
        auto u16ReadAt = [&](int64_t byteOffset) -> mlir::Value {
          mlir::Value ptr = rewriter.create<emitc::AddOp>(
              loc, qhBase.getType(), qhBase, sizeLit(byteOffset));
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                           u16ReadCallee, mlir::ValueRange{ptr})
              .getResult(0);
        };
        int64_t qhOffset =
            static_cast<int64_t>(qhBrick.getQhByteOffset().value_or(0));
        mlir::Value qhLow16 = u16ReadAt(qhOffset);
        mlir::Value qhHigh16 = u16ReadAt(qhOffset + 2);

        // The i32 sumi lvalue.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2). coreLmul is the
        // LOAD result LMUL (u8m1 for the weight; the vsetvl spelling is m1).
        auto loadVecType =
            llvm::cast<weftrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(flatPlan->blockLen)};
            });

        // HETEROGENEOUS loads: the WEIGHT strip is loaded U8 while the two q8
        // activation halves stay I8. The weight/activation quant offsets DIVERGE
        // (weight@8, acts@4/20) -- read straight off the load ops.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type u8CoreType =
            emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        mlir::Type u8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTailI8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        auto emitLoadTailU8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTailU8(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTailI8(yb, flatPlan->activationQuantOffset);
        valueMap[highLoad.getLoaded()] = emitLoadTailI8(
            yb, flatPlan->activationQuantOffset + flatPlan->blockLen);

        // The five-bit offset-binary nibble+qh decode + asymmetric widening
        // product, emitted from the product op's OWN weight/low/high operands (via
        // the valueMap) + the re-read qh halves + chunkOffset 0. applyOffsetBias
        // =FALSE is the ONE arithmetic delta vs q5_0: NO `vsub 16` is emitted (the
        // bias lives in the per-block MIN scale, folded through m_x*s_y).
        auto prodVecType = llvm::cast<weftrvv::VectorType>(
            fiveBitProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::Type u16WideType =
            emitc::OpaqueType::get(ctx, ("vuint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitFiveBitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(fiveBitProduct.getWeight()),
                valueMap.lookup(fiveBitProduct.getActivationLow()),
                valueMap.lookup(fiveBitProduct.getActivationHigh()), qhLow16,
                qhHigh16, chunkOffset, vl, i8CoreType, u8CoreType, u16WideType,
                i16WideType, coreLmul, wideLmul, 16, wideLmul, "i16", opName,
                role, flatPlan->applyOffsetBias);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[fiveBitProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand: seed a FRESH
        // literal-0 lane (per-block, no sumi carry), vwredsum the product, pull
        // lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // item4: deferred fp16 scale/min reads (fcvt.s.h now placed after the
        // integer core / right before the fold). block_base_x/y were emitted and
        // memoized above (shared with the loads + qh reads), so these blockBaseFor
        // calls HIT the memo -- ONLY the four (float)*(const _Float16 *) reads are
        // re-emitted here. Values byte-identical; only fcvt position moves.
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());
        mlir::Value mX = fp16ReadAt(
            blockBaseFor(
                minBrick.getLhsMinBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            minBrick.getLhsMinByteOffset());
        mlir::Value sY = fp16ReadAt(
            blockBaseFor(
                minBrick.getRhsSumBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            minBrick.getRhsSumByteOffset());

        // brick 1 ((d_x*d_y)*sumi) + the MIN brick (m_x*s_y) + brick 3 (sumf +
        // term) fold COLLECTIVELY into the one fused emitc.expression
        // (ScalePlusMin), fed the operand-derived d_x/d_y/m_x/s_y + the sumi
        // lvalue. The m_x*s_y mul is recomputed INSIDE the fused expression, so
        // the min brick's f32 result is gate-only.
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY, mX, sY);
      } else if (peekCodebookGather) {
        // ---- iq4_nl half-block asymmetric CODEBOOK-GATHER packed-i4 x i8 core
        // (the 2nd primitive class). M1 constructed-WEAK byte-exact increment (NOT
        // a flip): the region carries the codebook table broadcast + the
        // codebook-gather product bricks (both certified by the [L-8] allowlist),
        // and -- like the q4_0 (left_assoc) branch below -- the emit reuses the
        // SHARED emitFlatBlockCore (which drives emitFlatIntegerCore's
        // CodebookGatherNibble decode) + emitFlatFold that the monolithic iq4_nl
        // mbf1/elided m1 path drives, so the per-block body is byte-identical to
        // emitFlatBlockDot's ggml_iq4_nl_q8_0 instance. The codebook decl +
        // broadcast were emitted above (at the monolith's positions);
        // codebookValues threads the broadcast register into st. The operand-flow
        // gate (I7 fail-closed) ties the emitted core to the region content.
        // Lightweight region gate: the codebook-gather product's operands are the
        // region's per-block loads (a u8 packed-i4 weight + two plain-i8 q8 halves
        // off the ABI buffers) and its table operand is the region's broadcast; the
        // product feeds the reduce -> lane0 extract -> brick 2's sumi.
        auto weightLoad =
            peekCodebookGather.getWeight().getDefiningOp<weftrvv::LoadOp>();
        auto lowLoad =
            peekCodebookGather.getActivationLow().getDefiningOp<weftrvv::LoadOp>();
        auto highLoad =
            peekCodebookGather.getActivationHigh().getDefiningOp<weftrvv::LoadOp>();
        if (!peekCodebookTable || !weightLoad || !lowLoad || !highLoad ||
            coreLoads.size() != 3 || !coreExtract ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() ||
            !matchesFlatPlanHalfActivationOffsets(lowLoad, highLoad))
          return rewriter.notifyMatchFailure(
              loopBody,
              "full iq4_nl codebook flat block-dot body requires the region "
              "integer core: a 16-entry codebook table broadcast, three per-block "
              "loads (a u8 packed-i4 weight + two plain-i8 q8 halves off the ABI "
              "buffers, each carrying a quant_byte_offset), an asymmetric "
              "codebook-gather packed-i4 x i8 product, and a lane0 scalar extract");
        if (peekCodebookGather.getTable() != peekCodebookTable.getResult())
          return rewriter.notifyMatchFailure(
              peekCodebookGather,
              "iq4_nl codebook-gather product table operand must be the region's "
              "codebook table broadcast result");
        if (!coreReduce ||
            coreReduce.getInput() != peekCodebookGather.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "iq4_nl integer-core reduce input must be the codebook-gather "
              "product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "iq4_nl integer-core lane0 extract input must be the "
                           "reduce");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the codebook integer-core lane0 "
                      "extract result");

        // The final flat_* plan supplies the codebook decode and fold. Raw
        // geometry remains sourced from the loop/body operations, and the table
        // payload remains sourced from the typed broadcast operation.
        FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
        descriptor.qk = qk;
        descriptor.weightStride = loopBody.getWeightBlockStride();
        descriptor.activationStride = loopBody.getActivationBlockStride();
        descriptor.quantOffset =
            static_cast<int64_t>(*weightLoad.getQuantByteOffset());
        descriptor.activationQuantOffset = flatPlan->activationQuantOffset;
        descriptor.highOffset = flatPlan->blockLen;
        descriptor.hasCodebook = true;
        descriptor.codebook = peekCodebookTable.getCodebook();

        BlockDotFacts facts = *finalSchedule;
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), codebookValues, sizeType, opName, role);

        mlir::FailureOr<FlatBlockCore> core =
            emitFlatBlockCore(rewriter, loc, st, blockLoop.getInductionVar(), 0,
                              /*forceRobust=*/false);
        if (mlir::failed(core))
          return mlir::failure();
        emitFlatFold(rewriter, loc, st, core->sumiVar, core->dX, core->dY,
                     core->mX, core->sY);
      } else if (flatPlan->decodePrimitive ==
                 FlatDecodePrimitive::OffsetBinaryNibble) {
        // ---- q4_0 half-block asymmetric offset-binary packed-i4 x i8 core.
        // Lightweight region gate (I7 fail-closed): the region must carry the
        // asymmetric packed-i4 product, the THREE per-block loads (one packed-i4
        // weight + two plain-i8 q8 halves), and the lane0 extract feeding brick
        // 2's sumi. Then the descriptor-driven emit reuses the SAME
        // emitFlatBlockCore + emitFlatFold the monolithic q4_0 uses, so the body
        // is byte-identical (the OffsetBinaryNibble decode + LeftAssoc fold). ----
        weftrvv::PackedI4OffsetBinaryXI8ProductOp packedProduct;
        loopBody.getBody().walk(
            [&](weftrvv::PackedI4OffsetBinaryXI8ProductOp o) {
              packedProduct = o;
            });
        if (!packedProduct || coreLoads.size() != 3 || !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q4_0 flat block-dot body requires the region integer core: "
              "three per-block i8 loads (a packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric offset-binary packed-i4 x i8 product, and "
              "a lane0 scalar extract");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");

        // ---- W4 (q4_0 anti-bypass): OP-BY-OP OPERAND-DRIVEN packed-i4 core. The
        // integer core is NO LONGER re-derived from the loop-body attrs through
        // the monolithic emitFlatBlockCore套壳: the packed-i4 product is lowered
        // from ITS OWN weight/activation_low/activation_high operands (via the
        // valueMap), so mutating any product operand changes the emitted bytes.
        // Pre-W4 the packed product was收 as PRESENCE only and the core was rebuilt
        // from the LOADS' descriptor offsets, so swapping the product's act_high /
        // act_low / weight operand emitted the SAME bytes (NOT_GENUINE). Each of
        // the THREE loads is emitted from ITS load op's buffer/block_stride/
        // quant_byte_offset; the packed decode+widening product reads the three
        // load results the product NAMES; the reduce reads its product operand; the
        // lane0 extract reads its reduce operand; brick 1's two fp16 reads share the
        // per-block base. The decode arithmetic routes to the SAME
        // emitOffsetBinaryDecodeProductValue (vxor 0x88 / vsll / vsra sign-extend /
        // vwmul low / vwmacc high) the monolith's OffsetBinaryNibble strip runs, and
        // emitFlatFold emits the LeftAssoc fp32 fold -- so the body stays
        // byte-identical to the monolithic q4_0 mbf1 instance, only sourced from the
        // region ops. ----

        // Follow the packed product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight, q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh. Each must be a
        // per-block load reading the matching ABI buffer with a quant_byte_offset.
        auto weightLoad =
            packedProduct.getWeight().getDefiningOp<weftrvv::LoadOp>();
        auto lowLoad =
            packedProduct.getActivationLow().getDefiningOp<weftrvv::LoadOp>();
        auto highLoad =
            packedProduct.getActivationHigh().getDefiningOp<weftrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride() ||
            !matchesFlatPlanHalfActivationOffsets(lowLoad, highLoad))
          return rewriter.notifyMatchFailure(
              loopBody,
              "q4_0 packed-i4 product operands must be the region's per-block "
              "loads: a packed-i4 weight load off the weight ABI buffer + two "
              "plain-i8 q8 (low/high) activation loads off the activation ABI "
              "buffer, each carrying a block_stride + quant_byte_offset");

        // The standalone reduce consumes the packed product; the lane0 extract
        // consumes the reduce (brick 2's sumi = the extract is checked above). Any
        // break fails closed so the emit provably tracks the region chain.
        if (!coreReduce || coreReduce.getInput() != packedProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q4_0 integer-core reduce input must be the packed-i4 product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q4_0 integer-core lane0 extract input must be the "
                           "reduce");

        // The LeftAssoc fold descriptor. Only descriptor.foldModel + the shared
        // state (sumfVar / floatType / i32Type) are consumed by emitFlatFold; the
        // integer core is emitted op-by-op below, NOT via emitFlatBlockCore.
        FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
        BlockDotFacts facts = *finalSchedule;
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // Shared per-block base memo (byte-exact to the monolith's single
        // blockBaseValue): brick 1's fp16 reads + the per-block i8 loads that name
        // the same (%buffer, %block_index) SSA pair share ONE `base + ib*stride`.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        // block_base_x FROM the weight load; block_base_y FROM the low q8 load (the
        // high q8 load names the SAME activation base + block_index -> memo hit).
        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1: the two per-block fp16 -> f32 scale reads off the SHARED base
        // (memo hits, no re-emit), byte-exact to the monolith's fp16ReadAt.
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());

        // The i32 sumi lvalue: `int32_t sumi; sumi = 0;` (byte-exact BQL:5735).
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2) -- q4_0's nibble
        // half-block strip is qk/2 bytes (NOT the whole qk q8_0 uses). coreLmul is
        // the LOAD result LMUL (i8m1).
        auto loadVecType =
            llvm::cast<weftrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(flatPlan->blockLen)};
            });

        // The THREE per-block i8 loads, emitted in the monolith's order (weight ->
        // low q8 -> high q8), each from ITS load op's block base + quant_byte_offset
        // (+ the SHARED 0 chunk literal) + i8* cast + vle8 (byte-exact to
        // chunkPtr+loadI8). Keyed in the valueMap on the load result so the packed
        // product below reads them by the operand the product NAMES.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTail = [&](mlir::Value blockBase,
                                int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTail(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTail(yb, flatPlan->activationQuantOffset);
        valueMap[highLoad.getLoaded()] = emitLoadTail(
            yb, flatPlan->activationQuantOffset + flatPlan->blockLen);

        // The packed-i4 offset-binary decode + asymmetric widening product,
        // emitted from the packed product op's OWN weight/low/high operands (via
        // the valueMap) -- the SAME emitOffsetBinaryDecodeProductValue arithmetic
        // (vxor 0x88 / vsll / vsra sign-extend / vwmul low / vwmacc high) the
        // monolith's OffsetBinaryNibble strip reduce runs, so byte-identical while
        // the operands stay op-sourced. wideLmul is the product RESULT LMUL (i16m2).
        auto prodVecType =
            llvm::cast<weftrvv::VectorType>(packedProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(packedProduct.getWeight()),
                valueMap.lookup(packedProduct.getActivationLow()),
                valueMap.lookup(packedProduct.getActivationHigh()), vl,
                i8CoreType, i16WideType, "i8", coreLmul, 16, wideLmul, "i16",
                opName, role);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[packedProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand (byte-exact
        // BQL:5711-5766): seed a FRESH literal-0 lane (per-block, no sumi carry),
        // vwredsum the product, pull lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // brick 1 ((float)sumi * d_x) * d_y (LeftAssoc) + brick 3 (sumf + term)
        // fold COLLECTIVELY into the one fused emitc.expression, fed the
        // operand-derived d_x/d_y + the sumi lvalue. emitFlatFold's fold tree is
        // the formula-produced final flat fold plan.
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY,
                     /*mX=*/mlir::Value(), /*sY=*/mlir::Value());
      } else if (flatPlan->decodePrimitive == FlatDecodePrimitive::PlainI8) {

      // The q8_0 (plain_i8 / whole-block) descriptor projects the final flat_*
      // plan plus raw typed geometry. The int8 quant payload sits past the fp16 scale
      // header (quantOffset = stride - qk); q8_0 carries no qh / min / codebook.
      // (This else-branch full-body path is superseded by isQ80ScheduleParam for
      // any brick-2 body; kept in sync with the pinned SeparatedLeftAssoc oracle.)
      FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
      descriptor.qk = qk;
      descriptor.weightStride = loopBody.getWeightBlockStride();
      descriptor.activationStride = loopBody.getActivationBlockStride();
      descriptor.quantOffset = descriptor.weightStride - qk;

      // ---- W4: the REGION-DRIVEN integer-core chain gate. The full-body emit is
      // driven OP-BY-OP from these ops' operands (below); this gate first pins the
      // region as actually CARRYING the chain -- two per-block i8 loads -> signed
      // widening product -> standalone reduce -> lane0 scalar extract -> the
      // scalar i32 sumi feeding brick 2. Every link is checked against the
      // region SSA wiring; deleting or misdirecting ANY link fails closed here,
      // so the emit provably tracks the region content (pre-W4 the chain was
      // ignored and a broken/missing chain still emitted). ----
      if (coreLoads.size() != 2 || !coreProduct || !coreReduce || !coreExtract)
        return rewriter.notifyMatchFailure(
            loopBody,
            "full flat block-dot body requires the region integer core: two "
            "per-block i8 loads, a signed widening product, a standalone "
            "reduce, and a lane0 scalar extract");

      // Identify the weight vs activation per-block load by ABI buffer.
      weftrvv::LoadOp weightLoad, activationLoad;
      for (weftrvv::LoadOp ld : coreLoads) {
        if (ld.getBuffer() == loopBody.getWeightBase())
          weightLoad = ld;
        else if (ld.getBuffer() == loopBody.getActivationBase())
          activationLoad = ld;
      }
      if (!weightLoad || !activationLoad)
        return rewriter.notifyMatchFailure(
            loopBody, "integer-core loads must read the weight and activation "
                      "ABI buffers of the loop body");

      // Both loads are per-block, striding on the loop induction variable, with
      // the block stride / quant offset the descriptor scheduled (byte-exact to
      // the monolith's blockBaseValue+loadI8).
      for (auto [ld, stride, quantOff] :
           {std::make_tuple(weightLoad, descriptor.weightStride,
                            descriptor.quantOffset),
            std::make_tuple(activationLoad, descriptor.activationStride,
                            descriptor.activationQuantOffset)}) {
        if (ld.getBlockIndex() != coreBlock.getArgument(0))
          return rewriter.notifyMatchFailure(
              ld, "integer-core load block_index must be the loop induction "
                  "variable");
        if (ld.getBlockStride() != stride)
          return rewriter.notifyMatchFailure(
              ld, "integer-core load block_stride must match the scheduled "
                  "block stride");
        if (ld.getQuantByteOffset() != quantOff)
          return rewriter.notifyMatchFailure(
              ld, "integer-core load quant_byte_offset must match the "
                  "scheduled quant offset");
      }

      // The signed widening product multiplies the two per-block load results
      // (operand order free).
      if (coreProduct.getKind() != "signed_widening_product")
        return rewriter.notifyMatchFailure(
            coreProduct, "integer-core widening product must be a signed "
                         "widening product");
      mlir::Value pl = coreProduct.getLhs(), pr = coreProduct.getRhs();
      mlir::Value wLoaded = weightLoad.getLoaded(),
                  aLoaded = activationLoad.getLoaded();
      if (!((pl == wLoaded && pr == aLoaded) ||
            (pl == aLoaded && pr == wLoaded)))
        return rewriter.notifyMatchFailure(
            coreProduct, "integer-core widening product operands must be the "
                         "two per-block load results");

      // The standalone reduce consumes the widening product.
      if (coreReduce.getInput() != coreProduct.getResult())
        return rewriter.notifyMatchFailure(
            coreReduce,
            "integer-core reduce input must be the widening product");

      // The lane0 extract consumes the reduce.
      if (coreExtract.getInput() != coreReduce.getResult())
        return rewriter.notifyMatchFailure(
            coreExtract, "integer-core lane0 extract input must be the reduce");

      // The KEY link: brick 2's scalar i32 sumi must be the extracted lane0
      // scalar (the integer core -> scalar fold handoff), NOT a direct sumi.
      if (brick2.getSumi() != coreExtract.getResult())
        return rewriter.notifyMatchFailure(
            brick2, "brick 2 sumi must be the integer-core lane0 extract "
                    "result");

      // ---- W4: OP-BY-OP OPERAND-DRIVEN integer-core + scale emit. The full-body
      // integer core is NO LONGER re-derived from the loop-body attrs through the
      // monolithic emitFlatBlockCore套壳: every region op is lowered from ITS OWN
      // operands/attrs. The per-block base address flows from the LOAD op's
      // block_stride/quant_byte_offset (the address arithmetic tracks the load
      // operand, not the descriptor); the widening product reads its lhs/rhs load
      // results through the valueMap; the reduce reads its product operand; the
      // lane0 extract reads its reduce operand; brick 1's two fp16 reads read the
      // SHARED per-block base; and brick 1's d_x*d_y (+ brick 2 cast/mul + brick 3
      // add) fold COLLECTIVELY into the one fused emitc.expression. The emit stays
      // byte-identical to the monolithic q8_0 (plain_i8 / whole-block /
      // SumiTimesScales, m2, elided, mbf 1) instance (verified by golden diff), but
      // is now sourced from the region ops. Only emitFlatFold (the fused-tail
      // assembler, fed operand-derived d_x/d_y + the sumi lvalue) is reused for the
      // brick 1/2/3 collapse; the fold_model selecting the tree is the gated attr.
      BlockDotFacts facts = *finalSchedule;
      FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
          rewriter, descriptor, facts, weightBase, activationBase,
          sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
          opName, role);

      mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
      mlir::Value ib = blockLoop.getInductionVar();

      // Shared per-block base, key = (buffer SSA value, block_index SSA value).
      // brick 1's fp16-scale reads and the per-block i8 load BOTH name the same
      // (%buffer, %block_index) SSA pair, so ONE `base + ib*stride` (+ one step
      // comment) serves both -- byte-exact to the monolith's single blockBaseValue
      // build; emitting it twice would be a byte-diff.
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

      // block_base_x / block_base_y FROM the two per-block LOAD ops: the load's
      // block_stride drives the address mul (the operand-flow real gate).
      mlir::Value xb = blockBaseFor(
          weightLoad.getBuffer(), weightLoad.getBlockIndex(),
          static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
      mlir::Value yb = blockBaseFor(
          activationLoad.getBuffer(), activationLoad.getBlockIndex(),
          static_cast<int64_t>(*activationLoad.getBlockStride()), "block_base_y");

      // brick 1: the two per-block fp16 -> f32 scale reads, off the SHARED base
      // (memo hit, no re-emit). d_x = read(block_base_x + lhs_scale_byte_offset),
      // d_y = read(block_base_y + rhs_scale_byte_offset), byte-exact to fp16ReadAt.
      auto fp16ReadAt = [&](mlir::Value blockBase,
                            std::optional<int64_t> byteOffset) -> mlir::Value {
        mlir::Value addr = blockBase;
        if (byteOffset && *byteOffset != 0)
          addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                               blockBase, sizeLit(*byteOffset));
        return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                              mlir::ValueRange{addr}, opName, role,
                              llvm::StringRef("fcvt.s.h"));
      };
      mlir::Value dX = fp16ReadAt(
          blockBaseFor(
              brick1.getLhsScaleBase(), brick1.getBlockIndex(),
              static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
              "block_base_x"),
          brick1.getLhsScaleByteOffset());
      mlir::Value dY = fp16ReadAt(
          blockBaseFor(
              brick1.getRhsScaleBase(), brick1.getBlockIndex(),
              static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
              "block_base_y"),
          brick1.getRhsScaleByteOffset());

      // The i32 sumi lvalue: `int32_t sumi; sumi = 0;` -- the mutable target the
      // lane0 extract reassigns and the fold loads (byte-exact BQL:5735-5740).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The inner block-capped vl: ONE __riscv_vsetvl_e8<lmul>(qk) threaded as the
      // bodyVL to every core intrinsic. This is the ONE spot NOT sourced from the
      // region op operands -- the region ops carry the OUTER scope %vl, but the
      // byte-exact core caps the active vl at the qk sub-block. coreLmul is derived
      // from the LOAD's result vector type.
      auto loadVecType =
          llvm::cast<weftrvv::VectorType>(weightLoad.getLoaded().getType());
      llvm::StringRef coreLmul = loadVecType.getLmul();
      unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
      llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
      mlir::Value vl = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, innerSetvlCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            return {sizeLit(flatPlan->blockLen)};
          });

      // The two per-block i8 loads, each from ITS load op's block base + the
      // load's quant_byte_offset (+ the SHARED 0 chunk literal) + i8* cast + vle8.
      // The chunk-offset literal `0` is built ONCE and shared by both loads (the
      // monolith's single sizeLit(0) chunkOffset); only the quant offset literal
      // is fresh per load. The vle spelling is derived from the LOAD's LMUL.
      mlir::Type i8CoreType =
          emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
      mlir::Type i8PtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
      std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      mlir::Value chunkOffset = sizeLit(0);
      auto emitLoadTail = [&](mlir::Value blockBase,
                              int64_t quantOff) -> mlir::Value {
        mlir::Value withFixed = rewriter.create<emitc::AddOp>(
            loc, blockBase.getType(), blockBase, sizeLit(quantOff));
        mlir::Value full = rewriter.create<emitc::AddOp>(
            loc, blockBase.getType(), withFixed, chunkOffset);
        mlir::Value ptr =
            rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
        return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                              mlir::ValueRange{ptr, vl}, opName, role);
      };
      valueMap[weightLoad.getLoaded()] = emitLoadTail(
          xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
      valueMap[activationLoad.getLoaded()] = emitLoadTail(
          yb, flatPlan->activationQuantOffset);

      // The signed widening product, emitted from ITS lhs/rhs load-result
      // operands (via the valueMap) with the vwmul callee + widened type derived
      // from the RESULT vector type -- byte-exact to the monolith
      // `__riscv_vwmul_vv_i16m4(vx0, vy0, vl)` (lhs=weight first). The inner vl is
      // the bodyVL. emitWideningProduct would produce the identical call but stamp
      // the widening_product op's own provenance verbatim (a byte-diff vs the
      // monolith's loop-body provenance); hand-emitting here keeps the whole body
      // byte-identical while the operands + callee stay op-sourced.
      auto prodVecType =
          llvm::cast<weftrvv::VectorType>(coreProduct.getResult().getType());
      llvm::StringRef wideLmul = prodVecType.getLmul();
      mlir::Type i16WideType =
          emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
      std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
      valueMap[coreProduct.getResult()] = emitOpaqueCall(
          rewriter, loc, i16WideType, mulCallee,
          mlir::ValueRange{valueMap.lookup(coreProduct.getLhs()),
                           valueMap.lookup(coreProduct.getRhs()), vl},
          opName, role);

      // Reduce + lane0 extract, each from ITS op's input operand. The reduce seeds
      // a FRESH literal-0 lane (per-block, no sumi carry) then vwredsum's the
      // product; the extract pulls lane0 into the mutable sumi lvalue (byte-exact
      // BQL:5711-5766). The reduce/extract callees derive from the vector types.
      mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      mlir::Value seed = emitOpaqueCallBuilt(
          rewriter, loc, i32m1Type, seedCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value zero =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
            return {zero, sizeLit(1)};
          });
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      mlir::Value red = emitOpaqueCall(
          rewriter, loc, i32m1Type, reduceCallee,
          mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
          opName, role);
      valueMap[coreReduce.getResult()] = red;
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      mlir::Value extractVal = emitOpaqueCall(
          rewriter, loc, i32Type, extractCallee,
          mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
          role);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumi", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

      // brick 1 (d_x*d_y) + brick 2 ((float)sumi*scale) + brick 3 (acc+term) fold
      // COLLECTIVELY into the one fused emitc.expression (the single C statement
      // `sumf + (float)sumi * (d_x*d_y)`), fed the operand-derived d_x/d_y (brick 1
      // reads) + the sumi lvalue (from the extract). The acc operand (brick 3) is
      // the loop-carried sumf lvalue that emitFlatFold loads before the expression.
      emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY,
                   /*mX=*/mlir::Value(), /*sY=*/mlir::Value());
      } else if (flatPlan->decodePrimitive ==
                 FlatDecodePrimitive::UnsignedNibble) {
        // ---- q4_1 half-block asymmetric UNSIGNED-nibble packed-i4 x i8 core +
        // Family-B MIN correction. This is the q4_0 left_assoc op-by-op emit with
        // FOUR deltas: (1) the weight strip is loaded UNSIGNED (u8 / vle8_v_u8m1),
        // (2) the integer core routes to emitUnsignedNibbleDecodeProductValue
        // (vand 0x0F / vsrl 0x04 + reinterpret + vwmul/vwmacc) instead of the
        // offset-binary decode, (3) a per-block MIN brick reads m_x/s_y (fp16 at
        // byte offset 2 off the SHARED block bases), and (4) the fold is the
        // ScalePlusMin tree fed the harvested m_x/s_y. Byte-identical to the
        // monolithic q4_1 mbf1/elided instance; formula construction derives the
        // final decode/fold plan from the walked mechanisms and this branch
        // consumes it directly. ----
        weftrvv::UnsignedNibbleXI8ProductOp unsignedProduct;
        loopBody.getBody().walk(
            [&](weftrvv::UnsignedNibbleXI8ProductOp o) { unsignedProduct = o; });
        weftrvv::BlockFp16MinProductOp minBrick;
        loopBody.getBody().walk(
            [&](weftrvv::BlockFp16MinProductOp o) { minBrick = o; });
        if (!unsignedProduct || !minBrick || coreLoads.size() != 3 ||
            !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q4_1 flat block-dot body requires the region integer core: "
              "three per-block loads (a u8 packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric unsigned-nibble packed-i4 x i8 product, a "
              "lane0 scalar extract, and the per-block MIN brick");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");
        // The KEY Family-B link (I7 fail-closed): brick 2's min_term must be the
        // MIN brick's per-block m_x*s_y product (the scale_plus_min fold's second
        // product). Deleting or misdirecting the min brick fails closed here.
        if (brick2.getMinTerm() != minBrick.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 min_term must be the per-block MIN brick's "
                      "m_x*s_y correction product");

        // Follow the unsigned product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight (u8), q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh.
        auto weightLoad =
            unsignedProduct.getWeight().getDefiningOp<weftrvv::LoadOp>();
        auto lowLoad =
            unsignedProduct.getActivationLow().getDefiningOp<weftrvv::LoadOp>();
        auto highLoad =
            unsignedProduct.getActivationHigh().getDefiningOp<weftrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride() ||
            !matchesFlatPlanHalfActivationOffsets(lowLoad, highLoad))
          return rewriter.notifyMatchFailure(
              loopBody,
              "q4_1 unsigned-nibble product operands must be the region's "
              "per-block loads: a u8 packed-i4 weight load off the weight ABI "
              "buffer + two plain-i8 q8 (low/high) activation loads off the "
              "activation ABI buffer, each carrying a block_stride + "
              "quant_byte_offset");

        if (!coreReduce ||
            coreReduce.getInput() != unsignedProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q4_1 integer-core reduce input must be the unsigned-nibble "
              "product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q4_1 integer-core lane0 extract input must be the "
                           "reduce");

        // The ScalePlusMin fold descriptor. Only descriptor.foldModel + the
        // shared state are consumed by emitFlatFold; the integer core + the min
        // harvest are emitted op-by-op below, NOT via emitFlatBlockCore.
        FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
        BlockDotFacts facts = *finalSchedule;
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // Shared per-block base memo (byte-exact to the monolith's single
        // blockBaseValue): brick 1's + the MIN brick's fp16 reads and the
        // per-block loads that name the same (%buffer, %block_index) SSA pair
        // share ONE `base + ib*stride`.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1's two per-block fp16 -> f32 SCALE reads (d_x/d_y at offset 0),
        // off the SHARED base (memo hits, no re-emit), byte-exact to fp16ReadAt.
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        // item4 (fcvt.s.h reschedule): brick 1's d_x/d_y and the MIN brick's
        // m_x/s_y fp16->f32 reads are DEFERRED to after the integer core, right
        // before the fold (ggml factory placement). Only the fcvt.s.h reads move;
        // block_base_x/y (memoized, shared with the loads) and the fold
        // arithmetic/rounding stay put, so the emitted VALUES are byte-identical
        // (fold-oracle §1 preserved). The deferred reads are emitted just before
        // emitFlatFold below.

        // The i32 sumi lvalue.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2). coreLmul is the
        // LOAD result LMUL (u8m1 for the weight; the vsetvl spelling is m1).
        auto loadVecType =
            llvm::cast<weftrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(flatPlan->blockLen)};
            });

        // HETEROGENEOUS loads: the WEIGHT strip is loaded U8 (const uint8_t* /
        // __riscv_vle8_v_u8m1) while the two q8 activation halves stay I8 --
        // matching the monolith UnsignedNibble case (loadU8 weight + loadI8
        // y0/y1). If the weight were emitted i8, the u8 vand/vsrl callees below
        // would mismatch the operand type.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type u8CoreType =
            emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        mlir::Type u8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTailI8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        auto emitLoadTailU8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTailU8(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTailI8(yb, flatPlan->activationQuantOffset);
        valueMap[highLoad.getLoaded()] = emitLoadTailI8(
            yb, flatPlan->activationQuantOffset + flatPlan->blockLen);

        // The unsigned-nibble decode + asymmetric widening product, emitted from
        // the product op's OWN weight/low/high operands (via the valueMap) -- the
        // SAME emitUnsignedNibbleDecodeProductValue arithmetic the monolith's
        // UnsignedNibble strip reduce runs, so byte-identical while op-sourced.
        auto prodVecType = llvm::cast<weftrvv::VectorType>(
            unsignedProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitUnsignedNibbleDecodeProductValue(
                rewriter, loc, valueMap.lookup(unsignedProduct.getWeight()),
                valueMap.lookup(unsignedProduct.getActivationLow()),
                valueMap.lookup(unsignedProduct.getActivationHigh()), vl,
                i8CoreType, u8CoreType, i16WideType, coreLmul, 16, wideLmul,
                "i16", opName, role);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[unsignedProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand: seed a FRESH
        // literal-0 lane (per-block, no sumi carry), vwredsum the product, pull
        // lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // item4: deferred fp16 scale/min reads (fcvt.s.h now placed after the
        // integer core / right before the fold). block_base_x/y were emitted and
        // memoized above (shared with the loads), so these blockBaseFor calls HIT
        // the memo -- ONLY the four (float)*(const _Float16 *) reads are re-emitted
        // here. Values byte-identical; only fcvt position moves.
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());
        mlir::Value mX = fp16ReadAt(
            blockBaseFor(
                minBrick.getLhsMinBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            minBrick.getLhsMinByteOffset());
        mlir::Value sY = fp16ReadAt(
            blockBaseFor(
                minBrick.getRhsSumBase(), minBrick.getBlockIndex(),
                static_cast<int64_t>(minBrick.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            minBrick.getRhsSumByteOffset());

        // brick 1 ((d_x*d_y)*sumi) + the MIN brick (m_x*s_y) + brick 3 (sumf +
        // term) fold COLLECTIVELY into the one fused emitc.expression, fed the
        // operand-derived d_x/d_y/m_x/s_y + the sumi lvalue. emitFlatFold's fold
        // tree is the gated scale_plus_min fold_model; the m_x*s_y mul is
        // recomputed INSIDE the fused expression, so the min brick's f32 result is
        // gate-only (never materialized standalone).
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY, mX, sY);
      } else if (flatPlan->decodePrimitive ==
                     FlatDecodePrimitive::FiveBitOffsetBinary &&
                 flatPlan->applyOffsetBias) {
        // ---- q5_0 half-block asymmetric FIVE-BIT offset-binary packed-i4 (+ qh
        // 5th bit, `-16` bias) x i8 core + the ScalesTimesSumi fold. This is the
        // q4_1 op-by-op emit with FIVE deltas: (1) the integer core routes to
        // emitFiveBitOffsetBinaryDecodeProductValue (unsigned-nibble decode +
        // per-lane qh 5th-bit merge + `-16` offset-binary bias), (2) the 5th-bit
        // SOURCE is a per-block qh brick (re-read as two aligned u16 halves off the
        // SHARED weight base) sitting in the min-brick's slot -- NO m_x/s_y reads,
        // (3) the weight and activation quant offsets DIVERGE (weight qs@6 vs
        // activation qs@2/18, stamped by the front door), (4) the fold is the
        // ScalesTimesSumi tree `sumf + (d_x*d_y)*(float)sumi` with NO min term, and
        // (5) applyOffsetBias is true. Byte-identical to the monolithic q5_0
        // (five_bit_offset_binary / half-block / ScalesTimesSumi, m1, elided, mbf
        // 1) instance; formula construction derives the final decode/fold/bias
        // plan from the walked mechanisms and this branch consumes it directly. ----
        weftrvv::FiveBitOffsetBinaryXI8ProductOp fiveBitProduct;
        loopBody.getBody().walk(
            [&](weftrvv::FiveBitOffsetBinaryXI8ProductOp o) {
              fiveBitProduct = o;
            });
        weftrvv::BlockFiveBitQhSourceOp qhBrick;
        loopBody.getBody().walk(
            [&](weftrvv::BlockFiveBitQhSourceOp o) { qhBrick = o; });
        if (!fiveBitProduct || !qhBrick || coreLoads.size() != 3 || !coreExtract)
          return rewriter.notifyMatchFailure(
              loopBody,
              "full q5_0 flat block-dot body requires the region integer core: "
              "three per-block loads (a u8 packed-i4 weight + two plain-i8 q8 "
              "halves), an asymmetric five-bit offset-binary packed-i4 x i8 "
              "product, a lane0 scalar extract, and the per-block qh-source brick");
        if (brick2.getSumi() != coreExtract.getResult())
          return rewriter.notifyMatchFailure(
              brick2, "brick 2 sumi must be the integer-core lane0 extract "
                      "result");
        // q5_0's ScalesTimesSumi fold has NO min correction: brick 2 must NOT carry
        // a min_term (fail-closed against a mis-stamped q4_1/q5_1 body).
        if (brick2.getMinTerm())
          return rewriter.notifyMatchFailure(
              brick2, "q5_0 brick 2 must NOT carry a min_term (the "
                      "ScalesTimesSumi fold has no per-block MIN correction)");
        // The KEY five-bit link (I7 fail-closed): the product's qh_source operand
        // must be the per-block qh brick's gate-only token. Deleting or
        // misdirecting the qh brick fails closed here.
        if (fiveBitProduct.getQhSource() != qhBrick.getResult())
          return rewriter.notifyMatchFailure(
              fiveBitProduct, "q5_0 five-bit product qh_source must be the "
                              "per-block qh-source brick's gate-only token");

        // Follow the five-bit product's OPERANDS to their defining loads (the
        // operand-flow real gate): weight <- getWeight (u8), q8 low half <-
        // getActivationLow, q8 high half <- getActivationHigh.
        auto weightLoad =
            fiveBitProduct.getWeight().getDefiningOp<weftrvv::LoadOp>();
        auto lowLoad =
            fiveBitProduct.getActivationLow().getDefiningOp<weftrvv::LoadOp>();
        auto highLoad =
            fiveBitProduct.getActivationHigh().getDefiningOp<weftrvv::LoadOp>();
        if (!weightLoad || !lowLoad || !highLoad ||
            weightLoad.getBuffer() != loopBody.getWeightBase() ||
            lowLoad.getBuffer() != loopBody.getActivationBase() ||
            highLoad.getBuffer() != loopBody.getActivationBase() ||
            !weightLoad.getQuantByteOffset() || !lowLoad.getQuantByteOffset() ||
            !highLoad.getQuantByteOffset() || !weightLoad.getBlockStride() ||
            !lowLoad.getBlockStride() ||
            !matchesFlatPlanHalfActivationOffsets(lowLoad, highLoad))
          return rewriter.notifyMatchFailure(
              loopBody,
              "q5_0 five-bit product operands must be the region's per-block "
              "loads: a u8 packed-i4 weight load off the weight ABI buffer + two "
              "plain-i8 q8 (low/high) activation loads off the activation ABI "
              "buffer, each carrying a block_stride + quant_byte_offset");

        if (!coreReduce ||
            coreReduce.getInput() != fiveBitProduct.getResult())
          return rewriter.notifyMatchFailure(
              coreReduce ? coreReduce.getOperation() : loopBody.getOperation(),
              "q5_0 integer-core reduce input must be the five-bit product");
        if (coreExtract.getInput() != coreReduce.getResult())
          return rewriter.notifyMatchFailure(
              coreExtract, "q5_0 integer-core lane0 extract input must be the "
                           "reduce");

        // The ScalesTimesSumi fold descriptor. Only descriptor.foldModel + the
        // shared state are consumed by emitFlatFold; the integer core + the qh
        // harvest are emitted op-by-op below, NOT via emitFlatBlockCore.
        FlatBlockDotDescriptor descriptor = descriptorFromFinalPlan(*flatPlan);
        BlockDotFacts facts = *finalSchedule;
        FlatBlockDotEmitState st = buildFlatBlockDotEmitState(
            rewriter, descriptor, facts, weightBase, activationBase,
            sumfVar.getResult(), /*codebookValues=*/mlir::Value(), sizeType,
            opName, role);

        mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
        mlir::Value ib = blockLoop.getInductionVar();

        // Shared per-block base memo (byte-exact to the monolith's single
        // blockBaseValue): brick 1's + the qh brick's reads and the per-block loads
        // that name the same (%buffer, %block_index) SSA pair share ONE
        // `base + ib*stride`.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, mlir::Value>
            blockBaseMemo;
        auto blockBaseFor = [&](mlir::Value bufferSSA, mlir::Value blockIndexSSA,
                                int64_t stride, const char *step) -> mlir::Value {
          std::pair<mlir::Value, mlir::Value> key(bufferSSA, blockIndexSSA);
          auto it = blockBaseMemo.find(key);
          if (it != blockBaseMemo.end())
            return it->second;
          mlir::Value emittedBase = valueMap.lookup(bufferSSA);
          rewriter.create<emitc::VerbatimOp>(loc,
                                             stepComment(opName, role, step));
          mlir::Value off =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
          mlir::Value base = rewriter.create<emitc::AddOp>(
              loc, emittedBase.getType(), emittedBase, off);
          blockBaseMemo[key] = base;
          return base;
        };

        mlir::Value xb = blockBaseFor(
            weightLoad.getBuffer(), weightLoad.getBlockIndex(),
            static_cast<int64_t>(*weightLoad.getBlockStride()), "block_base_x");
        mlir::Value yb = blockBaseFor(
            lowLoad.getBuffer(), lowLoad.getBlockIndex(),
            static_cast<int64_t>(*lowLoad.getBlockStride()), "block_base_y");

        // brick 1's two per-block fp16 -> f32 SCALE reads (d_x/d_y at offset 0),
        // off the SHARED base (memo hits, no re-emit), byte-exact to fp16ReadAt.
        auto fp16ReadAt = [&](mlir::Value blockBase,
                              std::optional<int64_t> byteOffset) -> mlir::Value {
          mlir::Value addr = blockBase;
          if (byteOffset && *byteOffset != 0)
            addr = rewriter.create<emitc::AddOp>(loc, blockBase.getType(),
                                                 blockBase, sizeLit(*byteOffset));
          return emitOpaqueCall(rewriter, loc, floatType, kFp16ScaleReadCallee,
                                mlir::ValueRange{addr}, opName, role,
                                llvm::StringRef("fcvt.s.h"));
        };
        // item4 (fcvt.s.h reschedule): brick 1's d_x/d_y fp16->f32 reads are
        // DEFERRED to after the integer core, right before the fold (ggml factory
        // placement). Only the fcvt.s.h reads move; block_base_x/y (memoized,
        // shared with the loads + the qh reads) and the fold arithmetic/rounding
        // stay put, so the emitted VALUES are byte-identical (fold-oracle §1
        // preserved). The deferred reads are emitted just before emitFlatFold below.

        // The qh field's TWO aligned 16-bit halves, read BEFORE the sumi decl
        // (byte-exact to the monolith's qhLow16,qhHigh16 read). Off the SHARED
        // weight base (memo hit) at the qh brick's OWN qh_byte_offset -- mutating
        // the brick's qh_base operand or qh_byte_offset attr changes these emitted
        // addresses -> changes the bytes (anti-bypass); the descriptor qh offset is
        // NEVER read in this path. NOT fp16ReadAt: the qh read is a raw
        // `(uint16_t)*(const uint16_t *)` call to a u32 (no fcvt hint).
        mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
        llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
        mlir::Value qhBase = blockBaseFor(
            qhBrick.getQhBase(), qhBrick.getBlockIndex(),
            static_cast<int64_t>(qhBrick.getBlockStride().value_or(0)),
            "block_base_x");
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "qh_field"));
        auto u16ReadAt = [&](int64_t byteOffset) -> mlir::Value {
          mlir::Value ptr = rewriter.create<emitc::AddOp>(
              loc, qhBase.getType(), qhBase, sizeLit(byteOffset));
          return rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                           u16ReadCallee, mlir::ValueRange{ptr})
              .getResult(0);
        };
        int64_t qhOffset =
            static_cast<int64_t>(qhBrick.getQhByteOffset().value_or(0));
        mlir::Value qhLow16 = u16ReadAt(qhOffset);
        mlir::Value qhHigh16 = u16ReadAt(qhOffset + 2);

        // The i32 sumi lvalue.
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumi", opName, role));
        auto sumiVar = rewriter.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(i32Type),
            emitc::OpaqueAttr::get(ctx, ""));
        rewriter.create<emitc::AssignOp>(
            loc, sumiVar,
            rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

        // The inner block-capped vl: ONE vsetvl_e8<lmul>(qk/2). coreLmul is the
        // LOAD result LMUL (u8m1 for the weight; the vsetvl spelling is m1).
        auto loadVecType =
            llvm::cast<weftrvv::VectorType>(weightLoad.getLoaded().getType());
        llvm::StringRef coreLmul = loadVecType.getLmul();
        unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
        llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
        std::string innerSetvlCallee =
            riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");
        mlir::Value vl = emitOpaqueCallBuilt(
            rewriter, loc, sizeType, innerSetvlCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              return {sizeLit(flatPlan->blockLen)};
            });

        // HETEROGENEOUS loads: the WEIGHT strip is loaded U8 (const uint8_t* /
        // __riscv_vle8_v_u8m1) while the two q8 activation halves stay I8. The
        // weight/activation quant offsets DIVERGE (weight@6, acts@2/18) -- the
        // front door stamped them; this emit reads them straight off the load ops.
        mlir::Type i8CoreType =
            emitc::OpaqueType::get(ctx, ("vint8" + coreLmul + "_t").str());
        mlir::Type u8CoreType =
            emitc::OpaqueType::get(ctx, ("vuint8" + coreLmul + "_t").str());
        mlir::Type i8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
        mlir::Type u8PtrType =
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
        std::string i8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
        std::string u8LoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
        mlir::Value chunkOffset = sizeLit(0);
        auto emitLoadTailI8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, i8CoreType, i8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        auto emitLoadTailU8 = [&](mlir::Value blockBase,
                                  int64_t quantOff) -> mlir::Value {
          mlir::Value withFixed = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), blockBase, sizeLit(quantOff));
          mlir::Value full = rewriter.create<emitc::AddOp>(
              loc, blockBase.getType(), withFixed, chunkOffset);
          mlir::Value ptr =
              rewriter.create<emitc::CastOp>(loc, u8PtrType, full).getResult();
          return emitOpaqueCall(rewriter, loc, u8CoreType, u8LoadCallee,
                                mlir::ValueRange{ptr, vl}, opName, role);
        };
        valueMap[weightLoad.getLoaded()] = emitLoadTailU8(
            xb, static_cast<int64_t>(*weightLoad.getQuantByteOffset()));
        valueMap[lowLoad.getLoaded()] =
            emitLoadTailI8(yb, flatPlan->activationQuantOffset);
        valueMap[highLoad.getLoaded()] = emitLoadTailI8(
            yb, flatPlan->activationQuantOffset + flatPlan->blockLen);

        // The five-bit offset-binary nibble+qh decode + asymmetric widening
        // product, emitted from the product op's OWN weight/low/high operands (via
        // the valueMap) + the re-read qh halves + chunkOffset 0 (the elided
        // single-strip base) -- the SAME emitFiveBitOffsetBinaryDecodeProductValue
        // arithmetic the monolith's FiveBitOffsetBinary strip reduce runs, so
        // byte-identical while op-sourced. applyOffsetBias=true is the q5_0 `-16`
        // vsub (q5_1 shares the fn with false).
        auto prodVecType = llvm::cast<weftrvv::VectorType>(
            fiveBitProduct.getResult().getType());
        llvm::StringRef wideLmul = prodVecType.getLmul();
        mlir::Type i16WideType =
            emitc::OpaqueType::get(ctx, ("vint16" + wideLmul + "_t").str());
        mlir::Type u16WideType =
            emitc::OpaqueType::get(ctx, ("vuint16" + wideLmul + "_t").str());
        mlir::FailureOr<mlir::Value> productOr =
            emitFiveBitOffsetBinaryDecodeProductValue(
                rewriter, loc, valueMap.lookup(fiveBitProduct.getWeight()),
                valueMap.lookup(fiveBitProduct.getActivationLow()),
                valueMap.lookup(fiveBitProduct.getActivationHigh()), qhLow16,
                qhHigh16, chunkOffset, vl, i8CoreType, u8CoreType, u16WideType,
                i16WideType, coreLmul, wideLmul, 16, wideLmul, "i16", opName,
                role, flatPlan->applyOffsetBias);
        if (mlir::failed(productOr))
          return mlir::failure();
        valueMap[fiveBitProduct.getResult()] = *productOr;

        // Reduce + lane0 extract, each from ITS op's input operand: seed a FRESH
        // literal-0 lane (per-block, no sumi carry), vwredsum the product, pull
        // lane0 into the mutable sumi lvalue.
        mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
        std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
        mlir::Value seed = emitOpaqueCallBuilt(
            rewriter, loc, i32m1Type, seedCallee, opName, role,
            [&](mlir::OpBuilder &b,
                mlir::Location l) -> llvm::SmallVector<mlir::Value> {
              mlir::Value zero =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              return {zero, sizeLit(1)};
            });
        std::string reduceCallee =
            ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
        mlir::Value red = emitOpaqueCall(
            rewriter, loc, i32m1Type, reduceCallee,
            mlir::ValueRange{valueMap.lookup(coreReduce.getInput()), seed, vl},
            opName, role);
        valueMap[coreReduce.getResult()] = red;
        std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
        mlir::Value extractVal = emitOpaqueCall(
            rewriter, loc, i32Type, extractCallee,
            mlir::ValueRange{valueMap.lookup(coreExtract.getInput())}, opName,
            role);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, extractVal);

        // item4: deferred fp16 scale reads (fcvt.s.h now placed after the integer
        // core / right before the fold). block_base_x/y were emitted and memoized
        // above (shared with the loads + qh reads), so these blockBaseFor calls HIT
        // the memo -- ONLY the two (float)*(const _Float16 *) reads are re-emitted
        // here. Values byte-identical; only fcvt position moves.
        mlir::Value dX = fp16ReadAt(
            blockBaseFor(
                brick1.getLhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getLhsBlockStride().value_or(0)),
                "block_base_x"),
            brick1.getLhsScaleByteOffset());
        mlir::Value dY = fp16ReadAt(
            blockBaseFor(
                brick1.getRhsScaleBase(), brick1.getBlockIndex(),
                static_cast<int64_t>(brick1.getRhsBlockStride().value_or(0)),
                "block_base_y"),
            brick1.getRhsScaleByteOffset());

        // brick 1 ((d_x*d_y)*(float)sumi, ScalesTimesSumi) + brick 3 (sumf + term)
        // fold COLLECTIVELY into the one fused emitc.expression, fed the
        // operand-derived d_x/d_y + the sumi lvalue. NO min term (q5_0's fold has
        // no per-block MIN correction).
        emitFlatFold(rewriter, loc, st, sumiVar.getResult(), dX, dY,
                     /*mX=*/mlir::Value(), /*sY=*/mlir::Value());
      } else {
        return rewriter.notifyMatchFailure(
            loopBody, "step-5b full body lowers the sumi_times_scales (q8_0), "
                      "left_assoc (q4_0), scale_plus_min (q4_1 min-only / q5_1 "
                      "five-bit+min), and scales_times_sumi (q5_0) folds; the "
                      "other flat fold trees are later steps");
      }
    }
  }
  }

  // *s = sumf;  -- byte-exact to emitFlatBlockDot:5931-5945 (structured scalar
  // store through the output pointer).
  auto outPointer = llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
  if (!outPointer)
    return rewriter.notifyMatchFailure(loopBody,
                                       "loop-body output not a pointer");
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "store_s"));
  mlir::Value outIndex =
      rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
  emitc::SubscriptOp outSubscript =
      rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
  mlir::Value sumfFinal =
      rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
  rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

  return mlir::success();
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
