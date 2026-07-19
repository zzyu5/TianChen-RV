//===- RVVRepackStripWidthMaterialization.cpp ----------------------------===//
//
// Materializes the resource-aware repack strip width (half_lanes) onto the
// in-IR ggml Q4_0 x Q8_0 16x1-repacked GEMM/GEMV ops from the selected RVV
// -march. This is the N3-reframed PRIZE seam: a compiler-automatic SELECTION
// enters llama's q4_0 hot path. The repack lowering tiles each 16-block-as-lane
// group into weight_interleave/half_lanes strips of half_lanes e16m1 lanes, so
// the SAME typed kernel diverges by capability -- VLEN=128 -> half_lanes 8 ->
// two 8-lane halves, VLEN=256 -> half_lanes 16 -> one 16-lane strip. One
// capability fact (the guaranteed minimum VLEN) -> divergent emitted code, on a
// real llama.cpp repack kernel, with no hand-authored half_lanes fixture.
//
// The strip width is a TARGET-CAPABILITY-derived realization choice (the e16m1
// lane count the configured target's VLEN affords), not a plugin-selected dtype/
// route. Per core-invariants:
//   * I1 -- capability stays a first-class queryable object; the strip width is
//     derived from the VLEN capability fact READ off the in-IR provider op (the
//     typed minimum_vlen attr the probe layer materialized), not a local -march
//     re-parse.
//   * I4 -- the materialized half_lanes MIRRORS the in-IR capability fact; the
//     provider op's minimum_vlen (materialized once at the probe layer) is the
//     source this pass reads, the half_lanes op attribute is the mirror the
//     emitter reads.
//   * I5 -- the width is derived from the validated ISA tier (the -march /
//     isa-vector-hints evidence), never inferred from ABI strings, family names,
//     route ids, or fabricated config; the pass probes no hardware and consults
//     no clang/cmake/compile-run toolchain facts.
//   * I7 -- the width is clamped to whole strips of the 16-way interleave and
//     the dialect verifier pins half_lanes in {8, 16}, so a malformed width
//     fails closed at the op.
//
// The safety invariant the divergence rests on: the repack is 16-way interleaved
// (block_q4_0x16: 256 qs[] bytes = 16 blocks-as-lanes, byte i = block(i%16)
// offset(i/16)), so a 16-lane strip at VLEN=256 reads BYTE-IDENTICAL repacked
// data to the two 8-lane halves at VLEN=128. The stamp holds ONLY while the
// repack stays 16-way interleaved.
//
// SECOND divergence axis (ISA generation): the pre-ratification RVV0.7.1
// generation (XuanTie xtheadvector on the C920) has NO fractional LMUL, so the
// repack's default fractional core (i8mf2) is rejected by the XuanTie toolchain.
// On RVV0.7 the stamp pins the WHOLE-LMUL core anchor (integer_core_lmul = "m1":
// i8m1 -> i16m2 -> i32m4 -> f32m4) AND its mandatory ONE-16-lane-strip width
// (half_lanes = 16). The whole-LMUL core reads the SAME interleaved bytes (the
// i8m1 strip is 16 i8 lanes at VLEN=128), so it is numerically identical to the
// VLEN=256 fractional one-strip form; the win is using the whole LMUL the ISA
// supports. The RVV-generation fact is READ off the provider op's rvv_version
// fact (materialized once at the probe layer), not a march-string branch (I3).
//
//===----------------------------------------------------------------------===//

#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>

namespace weftrvv = ::weft::rvv;

namespace weft::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVREPACKSTRIPWIDTH
#include "Weft/Transforms/Passes.h.inc"

namespace {

// Stamps the capability-derived strip width (half_lanes) onto a repack monolithic
// op from the NAMED closed form getRVVRepackStripHalfLanes(minVLEN, weightInterleave)
// = min(vlen/16, interleave): 128 -> 8, 256 -> 16 (the [GAP-P1] family/width flip).
// The pre-ratification RVV0.7.1 generation (no fractional LMUL) pins the whole-LMUL
// one-16-lane-strip form instead (half_lanes = weightInterleave, integer_core_lmul
// "m1"). `vlenBits` is the provider minimum_vlen fact READ off the in-IR capability
// object by the caller (NOT a -march re-parse), so the width follows the capability.
//
// The closed form ALSO returns the ATTRIBUTION reason (capability_strip_width, NEVER a
// -march re-parse, NEVER a cost-model static_order); it is the PINNED 改判 record the
// committed judgment lit (rvv-repack-strip-width-capability-flip-closedform-judgment)
// documents + FileChecks the width flip against. It is NOT stamped as an op attribute
// here: these monolithic repack ops enforce a strict discardable-attr allowlist
// (fail-closed, I7), and the emitter reads the half_lanes ODS attr directly -- so the
// width is the load-bearing output and stamping stays byte-exact (no extra attrs).
template <typename RepackOp>
void stampRepackStripWidth(RepackOp repack, std::int64_t vlenBits, bool isRVV0p7) {
  if (isRVV0p7) {
    repack.setIntegerCoreLmul("m1");
    repack.setHalfLanes(repack.getWeightInterleave());
    return;
  }
  weft::plugin::rvv::RVVRepackStripHalfLanesChoice choice =
      weft::plugin::rvv::getRVVRepackStripHalfLanes(vlenBits,
                                                    repack.getWeightInterleave());
  if (choice.halfLanes > 0) // guaranteed VLEN >= 128 -> capability_strip_width.
    repack.setHalfLanes(choice.halfLanes);
}

class MaterializeRVVRepackStripWidthPass final
    : public impl::MaterializeRVVRepackStripWidthBase<
          MaterializeRVVRepackStripWidthPass> {
public:
  using impl::MaterializeRVVRepackStripWidthBase<
      MaterializeRVVRepackStripWidthPass>::MaterializeRVVRepackStripWidthBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // READ the guaranteed minimum VLEN off the in-IR RVV capability provider op
    // (the TYPED minimum_vlen fact the probe layer -- MaterializeRVVProbedCapability
    // Axes -- materialized ONCE from -march). This pass does NOT re-parse -march:
    // the divergence flows through the typed capability object (I1/I4), not a local
    // deriveMinimumVLEN(march) re-parse. The pass's -march/isa-vector-hints options
    // are retained ONLY so a conflicting -march can be threaded for the decisive
    // experiment (provider minimum_vlen=256 vs -march zvl128b) -- they are NOT
    // consulted for the width; the provider fact is authoritative.
    std::int64_t vlenBits =
        plugin::rvv::readRVVProviderMinimumVLEN(module).value_or(0);

    // READ the RVV ISA generation off the SAME provider op. The pre-ratification
    // RVV0.7.1 generation (XuanTie xtheadvector on the C920) has NO fractional
    // LMUL, so the repack's default fractional core (i8mf2) is NOT emittable there:
    // the whole-LMUL form (i8m1 -> i16m2 -> i32m4 -> f32m4) is required, which is
    // ONE 16-lane strip (half_lanes 16) per 16-block group. So on RVV0.7 the stamp
    // pins BOTH the whole-LMUL core anchor (integer_core_lmul = "m1") AND its
    // mandatory strip width (half_lanes = 16), overriding the VLEN-derived width.
    // RVV1.0 leaves integer_core_lmul unset (the emitter defaults to the fractional
    // "mf2" chain, byte-identical to HEAD).
    bool isRVV0p7 = plugin::rvv::readRVVProviderRVVVersion(module) ==
                    plugin::rvv::RVVVersion::RVV0p7;

    // A provider that guarantees no >= 128 minimum (no minimum_vlen fact, or a
    // constrained tier) yields no strip width: leave any hand-authored half_lanes
    // intact (no-clobber, mirroring the probed-axes materializer's empty skip).
    if (vlenBits < 128)
      return;

    module.walk([&](mlir::Operation *op) {
      // NOTE: the q4_0 16x1-repacked GEMM's monolithic op
      // (weft_rvv.repack_gemm_q4_0_q8_0) is RETIRED, as is the GEVM's
      // (weft_rvv.repack_gemv_q4_0_q8_0). Their resource-aware strip width /
      // whole-LMUL anchor is now derived + stamped at CONSTRUCTION time by the
      // repack front door (RVVLowerQuantContraction.cpp lowerToRepackGemm /
      // lowerToRepackGemv), directly on the typed
      // weft_rvv.typed_repack_gemm_loop_body / typed_repack_gemv_loop_body region --
      // the region STRUCTURE (numHalves accumulators / per-column fold bricks) is
      // baked to half_lanes, so it cannot be re-stamped post-hoc here. The
      // q4_1/q8_0 GEVMs + the q4_1/q4_K GEMMs below still carry their monolithic ops
      // and participate.
      // The FAMILY-B q4_1 repacked GEMV diverges on the SAME resource-aware strip
      // width axis (the block-as-lane layout is byte-identical in shape to q4_0's
      // 16-way interleave), so it participates in the same capability-driven
      // half_lanes / whole-LMUL stamp.
      if (auto gemv = llvm::dyn_cast<weftrvv::GgmlRepackGemvQ41Q81Op>(op)) {
        stampRepackStripWidth(gemv, vlenBits, isRVV0p7);
        return;
      }
      // The FAMILY-B q4_1 repacked GEMM (prefill) diverges on the SAME
      // resource-aware strip width axis as its GEMV sibling and the q4_0 GEMM
      // (the block-as-lane weight layout is byte-identical in shape), so it
      // participates in the same capability-driven half_lanes / whole-LMUL stamp.
      if (auto gemm = llvm::dyn_cast<weftrvv::GgmlRepackGemmQ41Q81Op>(op)) {
        stampRepackStripWidth(gemm, vlenBits, isRVV0p7);
        return;
      }
      // The FAMILY-A symmetric q8_0 repacked GEMV diverges on the SAME
      // resource-aware strip width axis (block_q8_0x16 is the SAME 16-way
      // block-as-lane interleave in shape as q4_0; only the weight lane is full
      // int8, never a nibble), so it participates in the same capability-driven
      // half_lanes / whole-LMUL stamp.
      if (auto gemv = llvm::dyn_cast<weftrvv::GgmlRepackGemvQ80Q80Op>(op)) {
        stampRepackStripWidth(gemv, vlenBits, isRVV0p7);
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVRepackStripWidthPass() {
  return std::make_unique<MaterializeRVVRepackStripWidthPass>();
}

} // namespace weft::transforms
