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
// STAGE C1 (this file, the in-IR BRIDGE): the pass now LOWERS a repack-SELECTED
// request to the REAL tcrv_rvv.repack_gemv_q4_0_q8_0 op and DECLARES the kernel's
// weight-layout requirement as an OUTPUT CONTRACT (tcrv_rvv.weight_layout_contract
// = "x16"). The compiler is the layout's CONSUMER + the contract's DECLARER; the
// plain->x16 weight MATERIALIZATION lives OUTSIDE the IR (the load-time / JIT
// producer, stages C3-C4). This is the layout-as-input / declared-contract model:
// the per-tensor limit does NOT dissolve, it RELOCATES to the system layer that
// owns the bytes.
//
// THE CRUX (stage B was Option (i): byte-identical block-dot on every path; C1
// flips the repack-SELECTED cell): the concrete repack target requires
// pre-interleaved block_q4_0x16 weights (stride 288, interleave 16) the abstract
// op's PLAIN stride-18 weights cannot supply. The bridge emits the repack op
// carrying the x16 facts + the contract, and REALIZES it ONLY where the target
// capability affords a valid e16m1 strip width (deriveRepackHalfLanes(minVLEN) in
// {8, 16} -- minVLEN >= 128). A repack-SELECTED request with no capability strip
// width (the q4_0-prefill cell at VLEN0: half_lanes 0) stays the deferred
// block-dot stub. The block-dot-SELECTED branch (q4_0@K1, q8_0, q4_K) is
// UNCHANGED + byte-identical. The repack-SELECTED cell DELIBERATELY changes (it
// emits the repack kernel) -- that is the POINT of C1; lit-verified, NOT run.
//
// SAFETY (NOT a latent miscompile): the emitted repack kernel reads x16 weights
// but the abstract op carries PLAIN weights, so the emit is correct ONLY when the
// contract is honored (x16 provided = stages C3-C4). The abstract
// GgmlQuantContractionOp has NO real producer (it is authored ONLY in lit
// fixtures; there is no rewriter.create of it in any real pass), so this repack
// op is reachable ONLY via lit, NEVER in the real llama.cpp pipeline. The bridge
// ASSERTS the layout; the system must make it true. NO e2e/perf claim is made.
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

#include <algorithm>
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

// The option-2 stage-C1 OUTPUT CONTRACT carrier (carrier A, the in-IR op attr).
// When the bridge realizes a repack-SELECTED request as the real repack-GEMV op
// it stamps tcrv_rvv.weight_layout_contract = "x16": the DECLARED requirement
// that the weight bytes the emitted kernel reads are in the block_q4_0x16 layout
// (the op's weight_block_stride = 288 contract). The compiler ASSERTS the layout;
// some later layer (the load-time / JIT producer, stages C3-C4) must make it
// true. The repack-GEMV verifier's attr allow-list accepts this bounded name
// (RVVDialectWideningOps.cpp GgmlRepackGemvQ40Q80Op::verify isAllowedAttr).
constexpr llvm::StringLiteral kWeightLayoutContractAttr =
    "tcrv_rvv.weight_layout_contract";

// The repacked weight 16-way interleave (block_q4_0x16: 16 weight rows per group
// occupy 16 distinct vector lanes). MIRRORS the op verifier's weight_interleave
// == 16 pin and deriveRepackHalfLanes's clamp input.
constexpr std::int64_t kWeightInterleave = 16;

// Derives the resource-aware e16m1 strip width (half_lanes) from the guaranteed
// minimum VLEN, the SAME pure rule MaterializeRVVRepackStripWidth uses
// (RVVRepackStripWidthMaterialization.cpp:78): half_lanes = min(vlen/16, 16),
// so 128 -> 8, 256 -> 16. Returns 0 when the evidence guarantees no >= 128
// minimum (an empty -march, or a constrained tier) -- the bridge then has NO
// capability-derived strip width and CANNOT form a well-formed x16 repack op, so
// it leaves the request as the deferred block-dot stub (the honest no-capability
// behavior, e.g. the q4_0-prefill-at-VLEN0 cell).
std::int64_t deriveRepackHalfLanes(std::int64_t vlenBits) {
  if (vlenBits < 128)
    return 0;
  return std::min<std::int64_t>(vlenBits / 16, kWeightInterleave);
}

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

    // STAGE C1 (the in-IR BRIDGE): when the selection is Repack AND the target
    // capability supplies a valid e16m1 strip width (minVLEN >= 128 => half_lanes
    // in {8, 16}), REALIZE the request as the real tcrv_rvv.repack_gemv_q4_0_q8_0
    // op carrying the block_q4_0x16 facts + the DECLARED weight_layout_contract =
    // "x16" (the OUTPUT CONTRACT). The block-dot-SELECTED branch (q4_0@K1, q8_0,
    // q4_K) is UNCHANGED -- it still emits the byte-identical block-dot body with
    // weight_layout_contract IMPLICITLY plain (the deferred-stub provenance). A
    // Repack-SELECTED request with NO capability strip width (the prefill cell at
    // VLEN0: half_lanes 0) CANNOT form a well-formed x16 repack op, so it stays
    // the deferred block-dot stub -- the bridge realizes x16 ONLY where the
    // capability fact actually affords the strip width.
    bool isRepack =
        selection.algorithm == pluginrvv::ContractionAlgorithm::Repack;
    std::int64_t halfLanes = deriveRepackHalfLanes(minVLEN);
    bool isRVV0p7 = pluginrvv::deriveRVVVersion(march, isaVectorHints) ==
                    pluginrvv::RVVVersion::RVV0p7;
    if (isRepack && halfLanes != 0)
      return lowerToRepackGemv(op, selection, halfLanes, isRVV0p7);

    return lowerToBlockDot(op, selection);
  }

  // STAGE C1 bridge: realize a repack-SELECTED, capability-afforded request as
  // the real tcrv_rvv.repack_gemv_q4_0_q8_0 op. It reconstructs the block_q4_0x16
  // x16 facts (stride 288, interleave 16, weight quant offset 32, activation
  // stride 34 / offset 2) the verifier pins, derives the resource-aware
  // half_lanes from the capability VLEN, and stamps the DECLARED OUTPUT CONTRACT
  // tcrv_rvv.weight_layout_contract = "x16". The op carries the SAME SSA weight
  // pointer the abstract op carried (the IR cannot tell a plain base from an x16
  // base; both are const uint8_t *) -- the contract is the bridge's ASSERTION
  // that some later layer (C3-C4) hands it x16 bytes. On RVV0.7.1 the whole-LMUL
  // core anchor (integer_core_lmul = "m1") with its mandatory ONE 16-lane strip
  // (half_lanes = 16) is pinned, mirroring MaterializeRVVRepackStripWidth; on
  // RVV1.0 integer_core_lmul is left unset (the fractional mf2 default).
  //
  // SAFETY (NOT a latent miscompile): the emitted kernel reads x16 weights but
  // the abstract op carries PLAIN weights, so this emit is correct ONLY when the
  // contract is honored. The abstract GgmlQuantContractionOp has NO real producer
  // (it is authored ONLY in lit fixtures; there is no rewriter.create of it in
  // any real pass), so this repack op is reachable ONLY via lit, NEVER in the
  // real llama.cpp pipeline. The bridge ASSERTS the layout; the system (C3-C4)
  // must make it true. This is NOT yet e2e-correct on plain weights.
  mlir::LogicalResult
  lowerToRepackGemv(tcrvrvv::GgmlQuantContractionOp op,
                    const pluginrvv::ContractionSelection &selection,
                    std::int64_t halfLanes, bool isRVV0p7) {
    mlir::OpBuilder builder(op);

    // On RVV0.7.1 the repack core is the WHOLE-LMUL chain (i8m1 -> i16m2 ->
    // i32m4 -> f32m4): no fractional LMUL, so the 16-block-as-lane group is ONE
    // 16-lane strip -- the verifier pins integer_core_lmul == "m1" <=> half_lanes
    // == 16. RVV1.0 leaves integer_core_lmul unset (the fractional mf2 default).
    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    mlir::StringAttr integerCoreLmul =
        isRVV0p7 ? builder.getStringAttr("m1") : mlir::StringAttr();

    auto repack = builder.create<tcrvrvv::GgmlRepackGemvQ40Q80Op>(
        op.getLoc(), op.getResult().getType(),
        /*weight_base=*/op.getWeightBase(),
        /*activation_base=*/op.getActivationBase(),
        /*output=*/op.getOutput(),
        /*element_count=*/op.getElementCount(),
        // The repack-GEMV INTERNALIZES the N loop, so it consumes the runtime
        // column count nc the abstract op ALWAYS carries (column_count) -- the
        // exact reason stage A always carries nc.
        /*column_count=*/op.getColumnCount(),
        /*vl=*/op.getVl(),
        /*kind=*/llvm::StringRef("ggml_repack_gemv_q4_0_q8_0"),
        /*scale_model=*/op.getScaleModel(),
        /*qk=*/static_cast<uint64_t>(op.getQk()),
        // The block_q4_0x16 x16 ABI facts the verifier pins (NOT the abstract
        // op's plain stride-18 facts -- this op reads the REPACKED layout the
        // contract declares).
        /*weight_block_stride=*/static_cast<uint64_t>(288),
        /*activation_block_stride=*/
        static_cast<uint64_t>(op.getActivationBlockStride()),
        /*weight_quant_byte_offset=*/static_cast<uint64_t>(32),
        /*activation_quant_byte_offset=*/
        static_cast<uint64_t>(op.getQuantByteOffset()),
        /*weight_interleave=*/static_cast<uint64_t>(kWeightInterleave),
        /*half_lanes=*/static_cast<uint64_t>(emittedHalfLanes),
        /*integer_core_lmul=*/integerCoreLmul);

    // The in-compiler decision audit (the same INERT provenance triple the
    // block-dot branch stamps) PLUS the stage-C1 DECLARED OUTPUT CONTRACT. The
    // materialization is "realized": the bridge has lowered the selection to a
    // real repack op (the x16 WEIGHT materialization is the system's, but the
    // in-IR realization of the kernel is done here, no longer deferred).
    repack->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    repack->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    repack->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    repack->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    op.getResult().replaceAllUsesWith(repack.getResult());
    op.erase();
    return mlir::success();
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
