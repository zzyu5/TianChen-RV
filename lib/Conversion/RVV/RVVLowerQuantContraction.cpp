//===- RVVLowerQuantContraction.cpp ---------------------------------------===//
//
// The option-2 stage-A front-of-pipeline pass that lowers the abstract,
// algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op to a CONCRETE contraction
// op, running BEFORE the EmitC lowering. In stage A this is a pure IDENTITY
// DEFAULT: every quant_contraction request lowers to the concrete
// tcrv_rvv.q4_0_q8_0_block_dot op with attributes reconstructed verbatim, so the
// IR after this pass is STRUCTURALLY IDENTICAL to today's hand-authored
// block-dot body and every downstream pass runs byte-for-byte unaffected.
//
// The block-dot identity branch DROPS the abstract op's column_count (nc)
// operand: the block-dot kernel (ggml_vec_dot_q4_0_q8_0) writes ONE fp32 and
// delegates the M/N loops to ggml's mul_mat caller (a bare 4-operand vec_dot).
// The abstract op carries nc ALWAYS so a later repack branch can reach it; here
// it is dropped to keep the emitted C byte-identical. NO schedule attrs are
// stamped -- those remain MaterializeRVVQ40Schedule's job downstream.
//
// This pass changes ZERO runtime behavior. It is structural scaffolding that
// makes "the compiler selects the contraction algorithm" expressible
// in-compiler in later stages. The repack lowering branch is a deliberate
// stage-C ERROR STUB here and is never reached on the wired block-dot path. On
// any module with no quant_contraction op the pass is a structural no-op.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_RVVLOWERQUANTCONTRACTION
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class RVVLowerQuantContractionPass final
    : public impl::RVVLowerQuantContractionBase<RVVLowerQuantContractionPass> {
public:
  using impl::RVVLowerQuantContractionBase<
      RVVLowerQuantContractionPass>::RVVLowerQuantContractionBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::WalkResult result =
        module.walk([&](tcrvrvv::GgmlQuantContractionOp op) -> mlir::WalkResult {
          if (mlir::failed(lowerOne(op)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  // Lower ONE abstract contraction request to a concrete contraction op. Stage A
  // wires only the q4_0-decode block-dot IDENTITY branch (DESIGN §2.2); every
  // other cell (the M>>1 prefill/GEMM regime, any non-q4_0 quant) is the
  // stage-C/repack lowering target and is a deliberate ERROR STUB here, so a
  // misrouted op fails fail-closed (I7) rather than silently taking block-dot.
  mlir::LogicalResult lowerOne(tcrvrvv::GgmlQuantContractionOp op) {
    // The single algorithm decision stage A makes: the IDENTITY DEFAULT to
    // block-dot, on the q4_0-decode cell ONLY. The selection-logic stage (B)
    // replaces this gate with a real capability/resource-aware choice and adds
    // the repack lowering (which needs the stage-C plain->x16 materialization).
    if (op.getQuant() == "q4_0" && op.getMRegime() == "decode")
      return lowerToBlockDot(op);

    return op.emitError()
           << "stage-A quant_contraction lowering wires ONLY the q4_0 decode "
              "block-dot identity branch; the (quant=\""
           << op.getQuant() << "\", m_regime=\"" << op.getMRegime()
           << "\") cell is the stage-C/repack lowering target and is not yet "
              "implemented (the repack branch needs the stage-C plain->x16 "
              "weight materialization)";
  }

  // The plain->plain IDENTITY branch: reconstruct today's hand-authored
  // tcrv_rvv.q4_0_q8_0_block_dot op verbatim, DROPPING column_count (nc). The
  // resulting IR is structurally identical to the hand-authored block-dot body,
  // so the downstream schedule + EmitC passes emit byte-identical C.
  mlir::LogicalResult lowerToBlockDot(tcrvrvv::GgmlQuantContractionOp op) {
    mlir::OpBuilder builder(op);

    // Operands: DROP column_count (nc) -- the block-dot vec_dot delegates M/N to
    // ggml's mul_mat caller and is a bare 4-operand-plus-vl op.
    auto blockDot = builder.create<tcrvrvv::GgmlBlockDotQ40Q80Op>(
        op.getLoc(), op.getResult().getType(),
        /*weight_base=*/op.getWeightBase(),
        /*activation_base=*/op.getActivationBase(),
        /*output=*/op.getOutput(),
        /*element_count=*/op.getElementCount(),
        /*vl=*/op.getVl(),
        // Attrs reconstructed verbatim from the abstract request's pinned facts.
        /*kind=*/llvm::StringRef("ggml_q4_0_q8_0_block_dot"),
        /*scale_model=*/op.getScaleModel(),
        /*qk=*/static_cast<uint64_t>(op.getQk()),
        /*weight_block_stride=*/
        static_cast<uint64_t>(op.getWeightBlockStride()),
        /*activation_block_stride=*/
        static_cast<uint64_t>(op.getActivationBlockStride()),
        /*quant_byte_offset=*/static_cast<uint64_t>(op.getQuantByteOffset()),
        /*activation_high_byte_offset=*/
        static_cast<uint64_t>(op.getActivationHighByteOffset()),
        // NO schedule knobs -- MaterializeRVVQ40Schedule stamps them downstream,
        // exactly as today.
        /*integer_core_lmul=*/::mlir::StringAttr(),
        /*multi_block_factor=*/::mlir::IntegerAttr(),
        /*strip_elision=*/::mlir::StringAttr());

    op.getResult().replaceAllUsesWith(blockDot.getResult());
    op.erase();
    return mlir::success();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createRVVLowerQuantContractionPass() {
  return std::make_unique<RVVLowerQuantContractionPass>();
}

} // namespace tianchenrv::transforms
