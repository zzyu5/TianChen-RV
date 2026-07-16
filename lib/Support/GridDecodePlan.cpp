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
/// lower it. Only the three iq2 grid siblings below are reachable today. iq1_s /
/// iq1_m / iq3_xxs / iq3_s are NOT registered -- registering them would make the
/// verifier accept IR that no emitter leaf can lower. Their required axis values
/// (GridSignPlane::TernaryDelta, GridFoldArith::DeltaGrid, GridEntryWidth::I32x4,
/// GridSignPlane::Ksigns128) are therefore deliberately ABSENT from the enums:
/// the shape they need is specified in docs/reports/2026-07-17-C4a-grid参数化.md,
/// not forward-declared as unexercised code.
///
/// The DERIVED grid + sign planes are emit-period `static const` tables named
/// here; their literals are emitted by the Conversion layer (NEVER op attrs).
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
