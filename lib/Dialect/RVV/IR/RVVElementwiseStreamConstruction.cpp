//===- RVVElementwiseStreamConstruction.cpp -----------------------------===//
//
// The ONE byte-exact construction of the streaming forward-elementwise FRONT-DOOR
// typed region, shared by the pre-emitc RVVElementwiseStreamFrontDoor pass (and any
// future in-emitc fallback). See the header for the MAP/REDUCE/ROTATE shape split;
// the region SHAPE and the downstream emit arithmetic
// (emitTypedElementwiseLoopBody) are identical to the hand-authored typed region,
// so the emitted C is byte-identical. Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#include "Weft/Dialect/RVV/IR/RVVElementwiseStreamConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace weft::rvv {

std::optional<ForwardElementwiseFacts>
lookupForwardElementwiseFacts(llvm::StringRef model) {
  // The 5 CONSTRUCTED forward operators + their loop SHAPE. The MAP family
  // (scale/silu) carries no loop-carried accumulator; the REDUCE family
  // (rms_norm/soft_max) carries a loop-carried f64/f64m1 accumulator; the ROTATE
  // family (rope) carries a loop-carried f32 theta recurrence. Any other model is
  // left abstract (fail-open skip), mirroring the dequant tq1_0/tq2_0 skip.
  if (model == "scale")
    return ForwardElementwiseFacts{"map", 3};
  if (model == "silu")
    return ForwardElementwiseFacts{"map", 3};
  if (model == "rms_norm")
    return ForwardElementwiseFacts{"reduce", 4};
  if (model == "soft_max")
    return ForwardElementwiseFacts{"reduce", 4};
  if (model == "rope")
    return ForwardElementwiseFacts{"rotate", 5};
  // The four forward SUPPORT ops -- all MAP family (no loop-carried accumulator):
  // add/mul are BINARY (x, y, z, n -> elementwise_binary_map), cpy/gelu are UNARY
  // (x, y, n -> elementwise_copy_map / elementwise_gelu_map).
  if (model == "add")
    return ForwardElementwiseFacts{"map", 4};
  if (model == "mul")
    return ForwardElementwiseFacts{"map", 4};
  if (model == "cpy")
    return ForwardElementwiseFacts{"map", 3};
  if (model == "gelu")
    return ForwardElementwiseFacts{"map", 3};
  return std::nullopt;
}

mlir::LogicalResult
constructTypedElementwiseLoopBody(mlir::RewriterBase &rewriter,
                                  GgmlForwardElementwiseOp fwdOp,
                                  const ForwardElementwiseFacts &facts) {
  mlir::Location loc = fwdOp.getLoc();
  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::StringRef model = fwdOp.getElementwiseModel();
  mlir::OperandRange operands = fwdOp.getAbiOperands();

  // Model-specific ABI operand layout (n is always LAST); op0/op1 are the two
  // leading buffers that also feed the loop op (buffer, scalar/output).
  mlir::Value op0 = operands[0];
  mlir::Value op1 = operands[1];
  mlir::Value n = operands[operands.size() - 1];

  mlir::Type indexType = rewriter.getIndexType();
  mlir::Type f64Type = rewriter.getF64Type();
  mlir::Type f32Type = rewriter.getF32Type();

  const bool isMap = facts.reduceMapModel == "map";
  const bool isReduce = facts.reduceMapModel == "reduce";
  const bool isRotate = facts.reduceMapModel == "rotate";

  // The loop-carried accumulator type (REDUCE/ROTATE only): rms_norm's f64 scalar,
  // soft_max's f64m1 WIDENING vector, rope's f32 theta recurrence.
  mlir::Type accType;
  if (model == "rms_norm")
    accType = f64Type;
  else if (model == "soft_max")
    accType = VectorType::get(ctx, f64Type, getRVVLMULM1());
  else if (model == "rope")
    accType = f32Type;

  {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(fwdOp);

    // The typed elementwise strip-loop op (buffer, scalar, n) + shape attrs. The
    // scale MAP anchors its in-place strip at m8 on the LOOP op too (matching the
    // hand-authored region); the other 4 carry strip_lmul only where the brick does.
    mlir::OperationState loopState(
        loc, TypedElementwiseLoopBodyOp::getOperationName());
    loopState.addOperands({op0, op1, n});
    loopState.addAttribute(
        "kind", rewriter.getStringAttr("typed_elementwise_loop_body"));
    loopState.addAttribute("reduce_map_model",
                           rewriter.getStringAttr(facts.reduceMapModel));
    loopState.addAttribute("element_sew", rewriter.getI64IntegerAttr(32));
    // The m8-strip MAP family (scale in-place + add/mul binary + cpy copy) anchors
    // its strip at m8 on the LOOP op too (matching the hand-authored region + the
    // brick); silu is m2-pinned and gelu is scalar, so neither carries a loop-op
    // strip knob.
    if (model == "scale" || model == "add" || model == "mul" || model == "cpy")
      loopState.addAttribute("strip_lmul", rewriter.getStringAttr("m8"));
    loopState.addRegion();
    auto loopBody =
        llvm::cast<TypedElementwiseLoopBodyOp>(rewriter.create(loopState));

    // The region entry block: (strip_index) for MAP, (strip_index, acc) for
    // REDUCE/ROTATE (the loop-carried-accumulator model).
    llvm::SmallVector<mlir::Type> argTypes{indexType};
    llvm::SmallVector<mlir::Location> argLocs{loc};
    if (isReduce || isRotate) {
      argTypes.push_back(accType);
      argLocs.push_back(loc);
    }
    mlir::Block *block = rewriter.createBlock(
        &loopBody.getBody(), loopBody.getBody().end(), argTypes, argLocs);
    mlir::Value stripIndex = block->getArgument(0);
    mlir::Value acc = (isReduce || isRotate) ? block->getArgument(1) : mlir::Value();
    rewriter.setInsertionPointToStart(block);

    // The per-strip CORE brick + the yield (no operand for MAP; the updated
    // accumulator / stepped theta for REDUCE/ROTATE).
    if (model == "scale") {
      mlir::OperationState core(loc, ElementwiseScaleMapOp::getOperationName());
      core.addOperands({op0, op1, n, stripIndex});
      core.addAttribute("kind", rewriter.getStringAttr("elementwise_scale_map"));
      core.addAttribute("strip_lmul", rewriter.getStringAttr("m8"));
      rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(loc, mlir::ValueRange{});
    } else if (model == "silu") {
      mlir::OperationState core(loc, ElementwiseSiluMapOp::getOperationName());
      core.addOperands({op0, op1, n, stripIndex});
      core.addAttribute("kind", rewriter.getStringAttr("elementwise_silu_map"));
      rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(loc, mlir::ValueRange{});
    } else if (model == "add" || model == "mul") {
      // The BINARY two-input map (add/mul): the abstract ABI is (lhs, rhs, output,
      // n); op0=lhs, op1=rhs, output=operands[2]. The brick carries all three
      // buffers + the byte-exact combiner selector binary_op.
      mlir::Value output = operands[2];
      mlir::OperationState core(loc, ElementwiseBinaryMapOp::getOperationName());
      core.addOperands({op0, op1, output, n, stripIndex});
      core.addAttribute("kind",
                        rewriter.getStringAttr("elementwise_binary_map"));
      core.addAttribute("binary_op", rewriter.getStringAttr(model));
      core.addAttribute("strip_lmul", rewriter.getStringAttr("m8"));
      rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(loc, mlir::ValueRange{});
    } else if (model == "cpy") {
      // The pass-through copy (cpy): abstract ABI (input, output, n); op0=input,
      // op1=output. The brick is the byte-exact load->store copy strip.
      mlir::OperationState core(loc, ElementwiseCopyMapOp::getOperationName());
      core.addOperands({op0, op1, n, stripIndex});
      core.addAttribute("kind", rewriter.getStringAttr("elementwise_copy_map"));
      core.addAttribute("strip_lmul", rewriter.getStringAttr("m8"));
      rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(loc, mlir::ValueRange{});
    } else if (model == "gelu") {
      // The scalar-libm tanh gelu (gelu): abstract ABI (input, output, n);
      // op0=input, op1=output. The brick lowers to the scalar per-element tanhf
      // loop (no strip knob).
      mlir::OperationState core(loc, ElementwiseGeluMapOp::getOperationName());
      core.addOperands({op0, op1, n, stripIndex});
      core.addAttribute("kind", rewriter.getStringAttr("elementwise_gelu_map"));
      rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(loc, mlir::ValueRange{});
    } else if (model == "rms_norm") {
      mlir::Value eps = operands[2];
      mlir::OperationState core(
          loc, ElementwiseRmsNormReduceCoreOp::getOperationName());
      core.addOperands({op0, op1, eps, n, stripIndex, acc});
      core.addAttribute(
          "kind", rewriter.getStringAttr("elementwise_rms_norm_reduce_core"));
      core.addAttribute("strip_lmul", rewriter.getStringAttr("m8"));
      core.addTypes({f64Type});
      core.addRegion(); // the OPTIONAL fused rms_norm->mul epilogue: 0 blocks =
                        // plain rms_norm (the unfused byte-identical path).
      mlir::Operation *coreOp = rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(
          loc, mlir::ValueRange{coreOp->getResult(0)});
    } else if (model == "soft_max") {
      mlir::Value maxV = operands[2];
      mlir::OperationState core(
          loc, ElementwiseSoftMaxReduceCoreOp::getOperationName());
      // brick operand order: (output, input, max, n, strip_index, acc).
      core.addOperands({op1, op0, maxV, n, stripIndex, acc});
      core.addAttribute(
          "kind", rewriter.getStringAttr("elementwise_soft_max_reduce_core"));
      core.addTypes({accType});
      mlir::Operation *coreOp = rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(
          loc, mlir::ValueRange{coreOp->getResult(0)});
    } else { // rope
      mlir::Value thetaBase = operands[2];
      mlir::Value thetaScale = operands[3];
      mlir::OperationState core(
          loc, ElementwiseRopeRotateCoreOp::getOperationName());
      // brick operand order: (input, output, theta_base, theta_scale, n,
      // pair_index, theta).
      core.addOperands({op0, op1, thetaBase, thetaScale, n, stripIndex, acc});
      core.addAttribute(
          "kind", rewriter.getStringAttr("elementwise_rope_rotate_core"));
      core.addTypes({f32Type});
      mlir::Operation *coreOp = rewriter.create(core);
      rewriter.create<TypedElementwiseLoopYieldOp>(
          loc, mlir::ValueRange{coreOp->getResult(0)});
    }
    (void)isMap;
  }
  rewriter.eraseOp(fwdOp);

  return mlir::success();
}

} // namespace weft::rvv
