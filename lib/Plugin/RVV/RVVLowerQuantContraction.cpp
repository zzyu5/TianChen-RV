//===- RVVLowerQuantContraction.cpp ---------------------------------------===//
//
// The option-2 front-of-pipeline pass that lowers the abstract,
// algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op to a CONCRETE contraction
// op, running BEFORE the EmitC lowering.
//
// STAGE B (this file): the pass makes "the COMPILER itself selects
// repack-vs-block-dot from capability facts" actually TRUE. Per walked
// quant_contraction request it derives the target VLEN from the pass's -march
// (deriveMinimumVLEN -- the SAME capability authority every other capability-gated
// pass uses; the op's advisory min_vlen attr is NOT the source), lifts the op's
// committed WHAT attrs (quant, m_regime) to plain enums, and calls the pure,
// branch-free, capability-fact-driven selectContractionAlgorithm. The decision is
// stamped on the lowered op as three INERT audit attrs (tcrv_rvv.*) so it is
// provable in-IR and lit-CHECKable.
//
// THE CRUX (resolved -- Option (i)): the concrete repack target
// (tcrv_rvv.repack_gemv_q4_0_q8_0) requires pre-interleaved block_q4_0x16 weights
// (stride 288, interleave 16) the abstract op's PLAIN stride-18 weights cannot
// supply -- that plain->x16 weight MATERIALIZATION is stage C. So stage B emits
// the BYTE-IDENTICAL block-dot body for BOTH the Repack-SELECTED and the
// BlockDot-SELECTED branches, differentiating ONLY via the inert audit attrs
// (Repack-selected => path_materialization = "deferred-stage-c"). The compiler
// MAKES + RECORDS the selection in-compiler (the N3 novelty); the emitted C is
// byte-identical on every path (NO e2e algorithm switch, NO perf claim); the e2e
// materialization is stage C.
//
// The block-dot identity branch DROPS the abstract op's column_count (nc)
// operand: the block-dot kernel (ggml_vec_dot_q4_0_q8_0) writes ONE fp32 and
// delegates the M/N loops to ggml's mul_mat caller (a bare 4-operand vec_dot). NO
// schedule attrs are stamped -- those remain MaterializeRVVQ40Schedule's job
// downstream. On any module with no quant_contraction op the pass is a no-op.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVContractionPathSelection.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;
namespace pluginrvv = ::tianchenrv::plugin::rvv;

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_RVVLOWERQUANTCONTRACTION
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// The inert audit-attr names the stage-B pass stamps on the lowered concrete op.
// They are pure provenance (no SEW/LMUL/policy/dataflow config) -- the EmitC
// emitter ignores them, exactly like the MaterializeRVVQ40Schedule pass's
// additive "tcrv_rvv.q4_0_schedule.*" trail -- so the emitted C is byte-identical
// with them present. The block-dot verifier's attribute allow-list is widened to
// accept this bounded namespace (RVVDialectWideningOps.cpp isAllowedBlockDotAttr).
constexpr llvm::StringLiteral kAlgorithmAttr = "tcrv_rvv.contraction_algorithm";
constexpr llvm::StringLiteral kReasonAttr = "tcrv_rvv.path_selection_reason";
constexpr llvm::StringLiteral kMaterializationAttr =
    "tcrv_rvv.path_materialization";

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
  // Lift the abstract op's committed WHAT attr to the selector's pure enum. The
  // verifier already pins quant=="q4_0"; the full enum is encoded so the selector
  // is the complete in-compiler static prior, but only q4_0 is reachable here.
  static mlir::FailureOr<pluginrvv::QuantType>
  liftQuant(tcrvrvv::GgmlQuantContractionOp op) {
    if (op.getQuant() == "q4_0")
      return pluginrvv::QuantType::Q4_0;
    if (op.getQuant() == "q8_0")
      return pluginrvv::QuantType::Q8_0;
    if (op.getQuant() == "q4_K")
      return pluginrvv::QuantType::Q4_K;
    return mlir::FailureOr<pluginrvv::QuantType>(
        op.emitError() << "stage-B contraction-path selection does not "
                          "recognize quant \""
                       << op.getQuant() << "\"");
  }

  static mlir::FailureOr<pluginrvv::MRegime>
  liftMRegime(tcrvrvv::GgmlQuantContractionOp op) {
    if (op.getMRegime() == "decode")
      return pluginrvv::MRegime::Decode;
    if (op.getMRegime() == "prefill")
      return pluginrvv::MRegime::Prefill;
    return mlir::FailureOr<pluginrvv::MRegime>(
        op.emitError() << "stage-B contraction-path selection does not "
                          "recognize m_regime \""
                       << op.getMRegime() << "\"");
  }

  // The IN-COMPILER selection: derive the target VLEN from the pass's -march (the
  // capability authority -- NOT the op's advisory min_vlen attr), lift the
  // committed WHAT axes, and ask the pure fact-driven selector which algorithm to
  // commit to. Both branches emit the byte-identical block-dot body (Option (i)),
  // differentiated only by the inert audit attrs -- so the emitted C is unchanged
  // on every cell and the repack EFFECT is honestly deferred to stage C.
  mlir::LogicalResult lowerOne(tcrvrvv::GgmlQuantContractionOp op) {
    mlir::FailureOr<pluginrvv::QuantType> quant = liftQuant(op);
    if (mlir::failed(quant))
      return mlir::failure();
    mlir::FailureOr<pluginrvv::MRegime> mRegime = liftMRegime(op);
    if (mlir::failed(mRegime))
      return mlir::failure();

    // The DERIVED capability fact: the guaranteed minimum VLEN of the configured
    // target, from the SAME plugin-local authority deriveHasZvl128b /
    // MaterializeRVVQ40Schedule consume (default -march "" => 0 => no capability
    // => block-dot, the honest no-capability behavior).
    std::int64_t minVLEN = pluginrvv::deriveMinimumVLEN(march, isaVectorHints);

    pluginrvv::ContractionSelection selection =
        pluginrvv::selectContractionAlgorithm(*quant, *mRegime, minVLEN);

    return lowerToBlockDot(op, selection);
  }

  // Option (i): emit the byte-identical tcrv_rvv.q4_0_q8_0_block_dot body for BOTH
  // the BlockDot-selected (realized) and the Repack-selected (deferred-stage-c)
  // cases, reconstructing today's hand-authored attrs verbatim and DROPPING
  // column_count (nc). The ONLY per-cell difference is the three inert audit attrs
  // recording the in-compiler decision. The emitted C is byte-identical to today
  // on every path; the repack op is materialized by stage C, never here.
  mlir::LogicalResult
  lowerToBlockDot(tcrvrvv::GgmlQuantContractionOp op,
                  const pluginrvv::ContractionSelection &selection) {
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

    // Stamp the in-compiler decision as INERT audit attrs (emitter-ignored
    // provenance, like tcrv_rvv.q4_0_schedule.*). Repack-selected records that
    // the repack DECISION is real but its weight materialization is deferred to
    // stage C; BlockDot-selected records the choice as fully realized.
    bool isRepack =
        selection.algorithm == pluginrvv::ContractionAlgorithm::Repack;
    blockDot->setAttr(kAlgorithmAttr,
                      builder.getStringAttr(isRepack ? "repack" : "block-dot"));
    blockDot->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    blockDot->setAttr(kMaterializationAttr,
                      builder.getStringAttr(isRepack ? "deferred-stage-c"
                                                     : "realized"));

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
