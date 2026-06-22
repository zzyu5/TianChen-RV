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
//     derived from the VLEN capability fact via the SAME plugin-local authority
//     (deriveMinimumVLEN) the probe->capability conversion uses.
//   * I4 -- the materialized half_lanes MIRRORS the plugin-local C++ authority;
//     the authority (deriveMinimumVLEN) is the source of truth, the op attribute
//     is the mirror the emitter reads.
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
// supports. The RVV-generation fact is derived through the SAME plugin-local
// authority (deriveRVVVersion), not a march-string branch in a core pass (I3).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVREPACKSTRIPWIDTH
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// Derives the resource-aware e16m1 strip width (half_lanes) from the guaranteed
// minimum VLEN: half_lanes = vlen/16 (the e16m1 lane count of a 16-bit-element
// vector), clamped to `weightInterleave` so the 16-block-as-lane group always
// tiles into whole strips. 128 -> 8, 256 -> 16, 512+ -> 16 (one 16-lane strip
// run at the available vl, upper lanes inactive). Returns 0 when the evidence
// guarantees no >= 128 minimum (the pass then leaves any authored width intact).
std::int64_t deriveRepackHalfLanes(std::int64_t vlenBits,
                                   std::int64_t weightInterleave) {
  if (vlenBits < 128 || weightInterleave <= 0)
    return 0;
  std::int64_t lanes = vlenBits / 16; // e16m1 lane count
  return std::min(lanes, weightInterleave);
}

class MaterializeRVVRepackStripWidthPass final
    : public impl::MaterializeRVVRepackStripWidthBase<
          MaterializeRVVRepackStripWidthPass> {
public:
  using impl::MaterializeRVVRepackStripWidthBase<
      MaterializeRVVRepackStripWidthPass>::MaterializeRVVRepackStripWidthBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Derive the guaranteed minimum VLEN ONCE from the selected -march (+ optional
    // probed isa/vector hints) through the SAME plugin-local authority the
    // probe->capability conversion uses. No toolchain-probe facts are consulted.
    std::int64_t vlenBits =
        plugin::rvv::deriveMinimumVLEN(march, isaVectorHints);

    // Derive the RVV ISA generation through the SAME plugin-local authority. The
    // pre-ratification RVV0.7.1 generation (XuanTie xtheadvector on the C920) has
    // NO fractional LMUL, so the repack's default fractional core (i8mf2) is NOT
    // emittable there: the whole-LMUL form (i8m1 -> i16m2 -> i32m4 -> f32m4) is
    // required, which is ONE 16-lane strip (half_lanes 16) per 16-block group. So
    // on RVV0.7 the stamp pins BOTH the whole-LMUL core anchor (integer_core_lmul
    // = "m1") AND its mandatory strip width (half_lanes = 16), overriding the
    // VLEN-derived width (deriveMinimumVLEN floors xtheadvector at 128, which
    // would otherwise derive 8). RVV1.0 leaves integer_core_lmul unset (the
    // emitter defaults to the fractional "mf2" chain, byte-identical to HEAD).
    bool isRVV0p7 = plugin::rvv::deriveRVVVersion(march, isaVectorHints) ==
                    plugin::rvv::RVVVersion::RVV0p7;

    // A march that guarantees no >= 128 minimum (empty march, or a constrained
    // tier) derives no strip width: leave any hand-authored half_lanes intact
    // (no-clobber, mirroring the probed-axes materializer's empty-derive skip).
    if (vlenBits < 128)
      return;

    module.walk([&](mlir::Operation *op) {
      if (auto gemm = llvm::dyn_cast<tcrvrvv::GgmlRepackGemmQ40Q80Op>(op)) {
        if (isRVV0p7) {
          gemm.setIntegerCoreLmul("m1");
          gemm.setHalfLanes(gemm.getWeightInterleave());
          return;
        }
        std::int64_t width =
            deriveRepackHalfLanes(vlenBits, gemm.getWeightInterleave());
        if (width > 0)
          gemm.setHalfLanes(width);
        return;
      }
      if (auto gemv = llvm::dyn_cast<tcrvrvv::GgmlRepackGemvQ40Q80Op>(op)) {
        if (isRVV0p7) {
          gemv.setIntegerCoreLmul("m1");
          gemv.setHalfLanes(gemv.getWeightInterleave());
          return;
        }
        std::int64_t width =
            deriveRepackHalfLanes(vlenBits, gemv.getWeightInterleave());
        if (width > 0)
          gemv.setHalfLanes(width);
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVRepackStripWidthPass() {
  return std::make_unique<MaterializeRVVRepackStripWidthPass>();
}

} // namespace tianchenrv::transforms
