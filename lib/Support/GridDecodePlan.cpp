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
/// iq1_m / iq3_xxs / iq3_s remain NOT registered for the SAME C4a reason: their
/// axis values (GridEntryWidth::I32x4, GridSignPlane::Ksigns128) stay ABSENT from
/// the enums until a row + front door + leaf + oracle land together.
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
