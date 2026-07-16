//===- GridDecodePlan.cpp - Grid-codebook repack decode plan registry -----===//
//
// See GridDecodePlan.h for the C4a [B-4] parameterization rationale.
//
//===----------------------------------------------------------------------===//

#include "Weft/Support/GridDecodePlan.h"

using namespace weft;

namespace {

/// The CLOSED grid decode registry -- the ONE authority that replaced THREE
/// hand-synced `!= "iq2_xxs" && != "iq2_xs" && != "iq2_s"` string chains (the
/// grid core verifier, the repack GEVM emitter, and the repack GEMM emitter).
///
/// A row does NOT weaken the gate: every caller rejects a decode_model absent
/// from this table, so the table IS the closed set ([D-1] unknown = reject).
///
/// HONESTY NOTE (what this registry does and does not claim): a row is
/// REACHABLE only if the front door can construct it AND an emitter leaf can
/// lower it. C4a registered ONLY the three iq2 grid siblings and deliberately
/// REFUSED iq1_s, because a row whose verifier accepts IR that no emitter leaf
/// can lower ships an inconsistency. C4a-2 landed iq1_s the ONLY legitimate way:
/// registry row + front door + BOTH emitter leaves + a byte-exact oracle, all at
/// once -- which is also what promoted GridSignPlane::TernaryDelta and
/// GridFoldArith::DeltaGrid from "specified in a report" to axis values with a
/// real row using them.
///
/// C4a-3 landed iq1_m the same four ways (row + front door + BOTH leaves + oracle),
/// promoting GridFoldArith::DeltaGridGroupSum. iq3_xxs / iq3_s remain NOT registered
/// for the SAME C4a reason: GridEntryWidth::I32x4 stays ABSENT from the enum until a
/// row + front door + leaf + oracle land together. (C4a-3 ALSO found that the second
/// axis value this comment used to promise for them -- GridSignPlane::Ksigns128 --
/// does not exist as a need at all: iq3_xxs's sign plane is the SAME ksigns_iq2xs
/// mechanism Signs64 already carries. See GridDecodePlan.h.)
///
/// The DERIVED grid + sign planes are emit-period `static const` tables named
/// here; their literals are emitted by the Conversion layer (NEVER op attrs).
///
constexpr GridDecodePlan kGridDecodePlans[] = {
    // iq2_xxs -- the FIRST grid sibling: flat 256-entry grid, u8 grid index,
    // SINGLE ls per sub-block, DERIVED signs64 +-1 plane.
    {
        /*decodeModel=*/"iq2_xxs",
        /*gridArrayName=*/"weft_iq2xxs_grid",
        /*signArrayName=*/"weft_iq2xxs_signs64",
        /*gridEntryCount=*/256,
        /*entryWidth=*/GridEntryWidth::I64x8,
        /*signPlane=*/GridSignPlane::Signs64,
        /*lsArity=*/GridLsArity::Single,
        /*foldArith=*/GridFoldArith::SignScaleEighth,
    },
    // iq2_xs -- DUAL ls, 512-entry grid, u16 9-bit index, DERIVED signs64.
    {
        /*decodeModel=*/"iq2_xs",
        /*gridArrayName=*/"weft_iq2xs_grid",
        /*signArrayName=*/"weft_iq2xs_signs64",
        /*gridEntryCount=*/512,
        /*entryWidth=*/GridEntryWidth::I64x8,
        /*signPlane=*/GridSignPlane::Signs64,
        /*lsArity=*/GridLsArity::Dual,
        /*foldArith=*/GridFoldArith::SignScaleEighth,
    },
    // iq2_s -- DUAL ls, 1024-entry grid, u16 assembled index, DIRECT signs256.
    {
        /*decodeModel=*/"iq2_s",
        /*gridArrayName=*/"weft_iq2s_grid",
        /*signArrayName=*/"weft_iq2s_signs256",
        /*gridEntryCount=*/1024,
        /*entryWidth=*/GridEntryWidth::I64x8,
        /*signPlane=*/GridSignPlane::Signs256,
        /*lsArity=*/GridLsArity::Dual,
        /*foldArith=*/GridFoldArith::SignScaleEighth,
    },
    // iq1_s (C4a-2) -- the TERNARY-DELTA grid sibling. 2048-entry grid (the
    // LARGEST; the 11-bit index is built from qs[l] | (((qh>>3l)&7)<<8) at REPACK
    // time and lands as a u16 strip), SAME I64x8 entry width as the iq2 rows (the
    // reason this was the lowest-risk 4th row), SINGLE ls per sub-block
    // (2*((qh>>12)&7)+1), NO sign plane (the grid bytes are already signed
    // ternary) and a per-sub-block +-1 delta from qh bit15 that rides the
    // sign-plane byte-offset slot as a DELTA strip. Its fold is the DeltaGrid
    // dual-accumulator shape, the ONLY grid row that reads the activation bsums.
    {
        /*decodeModel=*/"iq1_s",
        /*gridArrayName=*/"weft_iq1s_grid",
        /*signArrayName=*/"", // TernaryDelta: no sign table exists to name.
        /*gridEntryCount=*/2048,
        /*entryWidth=*/GridEntryWidth::I64x8,
        /*signPlane=*/GridSignPlane::TernaryDelta,
        /*lsArity=*/GridLsArity::Single,
        /*foldArith=*/GridFoldArith::DeltaGrid,
    },
    // iq1_m (C4a-3) -- iq1_s's ternary-grid sibling, and the row that shows what the
    // axes buy: FOUR of its five axis values were already in the enums. It gathers the
    // SAME 2048 ternary grid literals as iq1_s (ggml's ggml_vec_dot_iq1_m_q8_K literally
    // indexes `iq1s_grid`; the table is named separately here only because the emitter
    // decls are per-format), the SAME I64x8 entry width, the SAME TernaryDelta "no sign
    // plane, the grid byte IS the weight" decode, and -- REUSED from iq2_xs/iq2_s -- the
    // SAME Dual ls arity (ls1 groups 0-1 / ls2 groups 2-3, here 2*((sc>>..)&7)+1).
    //
    // The ONE genuinely new axis value is the fold: iq1_m's delta is per 8-element GROUP
    // (four INDEPENDENT +-1 per sub-block, from qh[l/2] bits 0x08/0x80) rather than iq1_s's
    // one per sub-block, and its delta term therefore needs the per-GROUP-of-8 activation
    // sum -- which block_q8_K's per-SIXTEEN bsums cannot express (two 8-groups inside one
    // bsums entry carry independent signs). So iq1_m reads NO bsums and accumulates the
    // group sum in-kernel. See GridFoldArith::DeltaGridGroupSum.
    //
    // Its repack layout is this line's design (ggml's block_iq1_m is 56 B of packed
    // qs/qh/scales with NO inline d at all -- the fp16 d is ASSEMBLED from four nibbles
    // scattered across the scales words, `(sc[0]>>12) | ((sc[1]>>8)&0x00f0) |
    // ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)`, which the repack does ONCE so the kernel sees
    // an ordinary inline fp16 d strip exactly like every other row).
    {
        /*decodeModel=*/"iq1_m",
        /*gridArrayName=*/"weft_iq1m_grid",
        /*signArrayName=*/"", // TernaryDelta: no sign table exists to name.
        /*gridEntryCount=*/2048,
        /*entryWidth=*/GridEntryWidth::I64x8,
        /*signPlane=*/GridSignPlane::TernaryDelta,
        /*lsArity=*/GridLsArity::Dual,
        /*foldArith=*/GridFoldArith::DeltaGridGroupSum,
    },
};

} // namespace

const GridDecodePlan *weft::lookupGridDecodePlan(llvm::StringRef decodeModel) {
  for (const GridDecodePlan &plan : kGridDecodePlans)
    if (plan.decodeModel == decodeModel)
      return &plan;
  return nullptr;
}

llvm::ArrayRef<GridDecodePlan> weft::getGridDecodePlans() {
  return llvm::ArrayRef<GridDecodePlan>(kGridDecodePlans);
}
