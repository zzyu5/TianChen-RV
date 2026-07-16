//===- GridDecodePlan.h - Grid-codebook repack decode plan registry -------===//
//
// The SINGLE fail-closed authority for the grid-codebook repack decode family.
//
// C4a ([B-4] "codebook parameterization not started" mechanism landing point).
//
// BEFORE: the grid table + sign plane were HARD-SELECTED inside the emitter by a
// closed `Iq2DualGridVariant::{Xs,S}` enum, and the legal decode_model set was a
// hand-copied `!= "iq2_xxs" && != "iq2_xs" && != "iq2_s"` string chain repeated
// in THREE places -- the grid core verifier (RVVDialectWideningOps.cpp), the
// repack GEVM emitter and the repack GEMM emitter (RVVToEmitCBlockQuantLinear.cpp).
// Three hand-synced copies of one closed set, and a grid table that could not be
// named without editing the emitter.
//
// AFTER: the decode axes ride as DATA on a registry entry, and all three sites
// consult lookupGridDecodePlan(). An UNREGISTERED decode_model yields nullptr =>
// REJECT. Parameterizing the plan does NOT open the legality gate: the registry
// IS the closed set, so [D-1] "unknown = reject" is preserved BY CONSTRUCTION
// rather than by three hand-synced string chains that can drift apart.
//
// This header lives in Support because WeftRVVDialect (verifier) and
// WeftConversionRVV (emitter) both link it, and Conversion -> Dialect is the only
// legal direction between those two.
//
// The FIXED grid + sign planes stay DERIVED emit-period `static const` tables
// (NEVER op attrs) -- the plan NAMES them, it does not carry their literals, so
// the signs64 op-attr blocker cannot recur.
//
// SCOPE / HONESTY: only decode models that are BOTH front-door-constructible and
// emitter-lowerable are registered (today: the three iq2 grid siblings + iq1_s).
// The enums below carry ONLY axis values those rows exercise. C4a-2 landed iq1_s
// FOUR-ways-at-once (registry row + front door + emitter leaves + byte-exact
// oracle), which is what promoted GridSignPlane::TernaryDelta and
// GridFoldArith::DeltaGrid from "specified in the C4a report" to real axis values
// with a real row using them -- C4a deliberately refused to forward-declare them
// while no row existed. The still-unregistered iq1_m / iq3_xxs / iq3_s need axis
// values (GridEntryWidth::I32x4, GridSignPlane::Ksigns128) that remain ABSENT for
// the SAME reason: this header never advertises a capability the emitter lacks.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_GRIDDECODEPLAN_H
#define WEFT_SUPPORT_GRIDDECODEPLAN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace weft {

/// The grid codebook's per-entry packing width. NOT a cosmetic label: the
/// gather's index EEW and the entry->grid-byte reinterpret both derive from it.
/// All registered rows are I64x8 today; the value is recorded per row because it
/// is a FACT of the format (checkable against ggml), not because the emitter
/// currently dispatches on it.
enum class GridEntryWidth {
  I64x8, ///< 8 packed int8 per uint64 entry (iq2_xxs / iq2_xs / iq2_s / iq1_s).
};

/// The sign plane the decode folds onto the gathered grid bytes. The Signs64 /
/// Signs256 rows differ ONLY in the DERIVED table's contents -- which is exactly
/// why the plan can name the table as data instead of the emitter branching on a
/// variant enum. TernaryDelta is NOT a table at all: it is the ABSENCE of a sign
/// plane (the iq1_s grid bytes are ALREADY signed ternary), so a TernaryDelta row
/// carries an EMPTY signArrayName and its weight sign-plane byte offset addresses
/// the per-sub-block +-1 DELTA strip instead of a sign SELECTOR strip.
enum class GridSignPlane {
  Signs64,  ///< 7-bit selector -> DERIVED 64-entry +-1 plane (iq2_xxs, iq2_xs).
  Signs256, ///< DIRECT 256-entry plane (iq2_s).
  TernaryDelta, ///< NO sign plane; signed ternary grid + per-sub-block +-1 (iq1_s).
};

/// How many ls scales each sub-block carries. Selects which emitter body leaf
/// lowers the row (the single-ls iq2_xxs leaf vs the dual-ls shared leaf).
enum class GridLsArity {
  Single, ///< one ls per sub-block (iq2_xxs, iq1_s).
  Dual,   ///< ls1 (groups 0-1) / ls2 (groups 2-3) (iq2_xs, iq2_s).
};

/// The END-OF-BLOCK float fold shape. This is the axis that makes iq1_s a
/// STRUCTURALLY different leaf rather than a parameter tweak of the iq2 leaves:
///
///   SignScaleEighth: sumf += cvt(sumi) * (d_x*d_y); the 0.125 is applied ONCE at
///     the STORE, and there is a SINGLE integer accumulator (iq2_xxs/xs/s).
///   DeltaGrid: sumf += (d_x*d_y) * (cvt(sumi) + 0.125f*cvt(sumi1)); TWO integer
///     accumulators (the grid dot AND the delta-bsum term), the 0.125 is applied
///     per-block to sumi1 ONLY, and there is NO trailing store-side factor. The
///     sumi1 term consumes the ACTIVATION bsums, which the iq2 rows never read
///     (iq1_s).
enum class GridFoldArith {
  SignScaleEighth, ///< single i32 acc, trailing 0.125 at store (iq2_xxs/xs/s).
  DeltaGrid,       ///< sumi + 0.125*sumi1 dual-acc bsums delta fold (iq1_s).
};

/// One registered grid decode model. The plan is pure DATA: it NAMES the DERIVED
/// emit-period tables (never their literals) and pins the decode axes.
struct GridDecodePlan {
  llvm::StringRef decodeModel;   ///< the core brick's decode_model attr value.
  llvm::StringRef gridArrayName; ///< e.g. "weft_iq2xs_grid".
  /// e.g. "weft_iq2xs_signs64"; EMPTY iff signPlane == TernaryDelta (no table).
  llvm::StringRef signArrayName;
  std::int64_t gridEntryCount;   ///< 256 / 512 / 1024 / 2048.
  GridEntryWidth entryWidth;
  GridSignPlane signPlane;
  GridLsArity lsArity;
  GridFoldArith foldArith;
};

/// The CLOSED registry lookup -- the single fail-closed authority the grid core
/// VERIFIER and both grid EMITTER leaves consult. Returns nullptr for an
/// unregistered decode_model, which every caller MUST treat as REJECT ([D-1]).
const GridDecodePlan *lookupGridDecodePlan(llvm::StringRef decodeModel);

/// The registered plans, in registration order. Used to RENDER the verifier's
/// diagnostic so the accepted set can never drift from the registry.
llvm::ArrayRef<GridDecodePlan> getGridDecodePlans();

} // namespace weft

#endif // WEFT_SUPPORT_GRIDDECODEPLAN_H
