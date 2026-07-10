//===- RVVQuantizeRowConstruction.cpp -----------------------------------===//
//
// The ONE byte-exact construction of the streaming quantize_row FRONT-DOOR typed
// region, shared by the pre-emitc RVVQuantizeRowStreamFrontDoor pass and the
// in-emitc constructQuantizeRowRegionAndLower fallback. See the header for the
// design split (only WHEN the region is built differs; the region SHAPE and the
// downstream emit arithmetic are identical, so both paths emit byte-exact C).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"

namespace tianchenrv::tcrv::rvv {

std::optional<QuantizeRowStreamFacts>
lookupQuantizeRowStreamFacts(llvm::StringRef encodeModel) {
  // The 3 CONSTRUCTED streaming quantize_row encode_models + their per-format AoS
  // block-layout facts (the ggml ABI shape constants, NOT tunable knobs): qk (32
  // flat / 256 K-quant), the block stride, the scale byte offset, and the packed-
  // quant (int8 qs) byte offset. The remaining per-format offsets (the q8_1 fp16
  // block sum `s` @2; the q8_K per-16 int16 bsums @260) are baked into the per-
  // format encode leaf at emit, so the brick carries only the two byte offsets the
  // shared attr surface names.
  if (encodeModel == "q8_0") {
    // block_q8_0 (ggml-common.h:241-245 + QK8_0 = 32): fp16 d @0, 32 int8 qs @2,
    // stride 34.
    return QuantizeRowStreamFacts{/*qk=*/32, /*stride=*/34, /*scaleOff=*/0,
                                  /*quantOff=*/2};
  }
  if (encodeModel == "q8_1") {
    // block_q8_1 (ggml-common.h:248-259 + QK8_1 = 32): fp16 d @0, fp16 s @2, 32
    // int8 qs @4, stride 36 (the sum offset 2 is baked into the q8_1 encode leaf).
    return QuantizeRowStreamFacts{/*qk=*/32, /*stride=*/36, /*scaleOff=*/0,
                                  /*quantOff=*/4};
  }
  if (encodeModel == "q8_K") {
    // block_q8_K (ggml-common.h:360-366 + QK_K = 256): FLOAT d @0, 256 int8 qs @4,
    // 16 int16 bsums @260, stride 292 (the bsums offset 260 is baked into the q8_K
    // encode leaf).
    return QuantizeRowStreamFacts{/*qk=*/256, /*stride=*/292, /*scaleOff=*/0,
                                  /*quantOff=*/4};
  }
  // Any other format is NOT a constructed streaming quantizer; the front door does
  // not construct it.
  return std::nullopt;
}

mlir::LogicalResult constructTypedQuantizeRowLoopBody(
    mlir::RewriterBase &rewriter, mlir::Operation *quantOp, mlir::Value input,
    mlir::Value output, mlir::Value n, llvm::StringRef encodeModel,
    const QuantizeRowStreamFacts &facts) {
  mlir::Location loc = quantOp->getLoc();
  mlir::Type indexType = rewriter.getIndexType();

  {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(quantOp);

    mlir::OperationState loopState(
        loc, TypedQuantizeRowLoopBodyOp::getOperationName());
    loopState.addOperands({input, output, n});
    loopState.addAttribute(
        "kind", rewriter.getStringAttr("typed_quantize_row_loop_body"));
    loopState.addAttribute("qk", rewriter.getI64IntegerAttr(facts.qk));
    loopState.addAttribute("block_stride",
                           rewriter.getI64IntegerAttr(facts.blockStride));
    loopState.addAttribute("encode_model", rewriter.getStringAttr(encodeModel));
    loopState.addRegion();
    auto loopBody =
        llvm::cast<TypedQuantizeRowLoopBodyOp>(rewriter.create(loopState));

    mlir::Block *block = rewriter.createBlock(
        &loopBody.getBody(), loopBody.getBody().end(), {indexType}, {loc});
    mlir::Value blockIndex = block->getArgument(0);
    rewriter.setInsertionPointToStart(block);

    mlir::OperationState coreState(
        loc, QuantizeRowEncodeCoreOp::getOperationName());
    coreState.addOperands({input, output, blockIndex});
    coreState.addAttribute("encode_model", rewriter.getStringAttr(encodeModel));
    coreState.addAttribute("qk", rewriter.getI64IntegerAttr(facts.qk));
    coreState.addAttribute("block_stride",
                           rewriter.getI64IntegerAttr(facts.blockStride));
    coreState.addAttribute("scale_byte_offset",
                           rewriter.getI64IntegerAttr(facts.scaleByteOffset));
    coreState.addAttribute("quant_byte_offset",
                           rewriter.getI64IntegerAttr(facts.quantByteOffset));
    rewriter.create(coreState);
    rewriter.create<TypedQuantizeRowLoopYieldOp>(loc);
  }
  rewriter.eraseOp(quantOp);

  return mlir::success();
}

} // namespace tianchenrv::tcrv::rvv
