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
// emitter-lowerable are registered (today: the three iq2 grid siblings). The
// enums below carry ONLY axis values those rows exercise. The additional axis
// values iq1_s / iq1_m / iq3_xxs / iq3_s require (a no-sign-plane ternary-delta
// shape, a two-accumulator delta fold, and a 4-byte grid entry width) are NOT
// forward-declared here; that shape is specified in
// docs/reports/2026-07-17-C4a-grid参数化.md, so this header never advertises a
// capability the emitter does not have.
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
  I64x8, ///< 8 packed int8 per uint64 entry (iq2_xxs / iq2_xs / iq2_s).
};

/// The sign plane the decode folds onto the gathered grid bytes. The repack
/// stage NORMALIZES the sign selector, so these two rows differ ONLY in the
/// DERIVED table's contents -- which is exactly why the plan can name the table
/// as data instead of the emitter branching on a variant enum.
enum class GridSignPlane {
  Signs64,  ///< 7-bit selector -> DERIVED 64-entry +-1 plane (iq2_xxs, iq2_xs).
  Signs256, ///< DIRECT 256-entry plane (iq2_s).
};

/// How many ls scales each sub-block carries. Selects which emitter body leaf
/// lowers the row (the single-ls iq2_xxs leaf vs the dual-ls shared leaf).
enum class GridLsArity {
  Single, ///< one ls per sub-block (iq2_xxs).
  Dual,   ///< ls1 (groups 0-1) / ls2 (groups 2-3) (iq2_xs, iq2_s).
};

/// One registered grid decode model. The plan is pure DATA: it NAMES the DERIVED
/// emit-period tables (never their literals) and pins the decode axes.
struct GridDecodePlan {
  llvm::StringRef decodeModel;   ///< the core brick's decode_model attr value.
  llvm::StringRef gridArrayName; ///< e.g. "weft_iq2xs_grid".
  llvm::StringRef signArrayName; ///< e.g. "weft_iq2xs_signs64".
  std::int64_t gridEntryCount;   ///< 256 / 512 / 1024.
  GridEntryWidth entryWidth;
  GridSignPlane signPlane;
  GridLsArity lsArity;
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
