//===- RVVQuantizeRowConstruction.cpp -----------------------------------===//
//
// Mechanical realization of one already-constructed streaming quantize_row formula
// result.  See the header for the single pre-emission authority boundary.
//
//===----------------------------------------------------------------------===//

#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"

namespace weft::rvv {

mlir::LogicalResult constructTypedQuantizeRowLoopBody(
    mlir::RewriterBase &rewriter, mlir::Operation *quantOp, mlir::Value input,
    mlir::Value output, mlir::Value n, const QuantizeRowStreamFacts &facts) {
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
    loopState.addAttribute("quantize_leaf",
                           QuantizeRowLeafAttr::get(rewriter.getContext(),
                                                    facts.leaf));
    loopState.addAttribute("encode_model",
                           rewriter.getStringAttr(facts.encodeModel));
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
    coreState.addAttribute("quantize_leaf",
                           QuantizeRowLeafAttr::get(rewriter.getContext(),
                                                    facts.leaf));
    coreState.addAttribute("encode_model",
                           rewriter.getStringAttr(facts.encodeModel));
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

} // namespace weft::rvv
