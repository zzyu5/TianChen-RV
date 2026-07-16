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
// emitter-lowerable are registered (today: the three iq2 grid siblings + iq1_s +
// iq1_m + iq3_xxs). The enums below carry ONLY axis values those rows exercise.
// C4a-2 landed iq1_s FOUR-ways-at-once (registry row + front door + emitter leaves +
// byte-exact oracle), which is what promoted GridSignPlane::TernaryDelta and
// GridFoldArith::DeltaGrid from "specified in the C4a report" to real axis values
// with a real row using them -- C4a deliberately refused to forward-declare them
// while no row existed. C4a-3 landed iq1_m the same four ways, promoting
// GridFoldArith::DeltaGridGroupSum the same way; iq1_m REUSED three existing axis
// values unchanged (I64x8, TernaryDelta, and -- from the iq2 dual rows -- Dual), so
// the marginal enum cost of the 5th row was ONE value.
//
// C4a-4 landed iq3_xxs the same four ways, promoting GridEntryWidth::I32x4. What that
// row cost, and what it did NOT cost, is recorded at the two enums it touches, because
// BOTH of this header's prior predictions about it were WRONG and the corrections are
// the load-bearing part:
//
//   * WRONG PREDICTION #1 (retired by C4a-3, against ggml): "iq3_xxs needs a new
//     GridSignPlane::Ksigns128". It does not. Its sign plane is the ksigns_iq2xs
//     mechanism Signs64 already carries, and the landed row REUSES Signs64 unchanged.
//   * WRONG PREDICTION #2 (retired by C4a-4, BY LANDING THE ROW): "the still-
//     unregistered iq3_xxs / iq3_s NEED GridEntryWidth::I32x4" -- stated as though the
//     enum value were the blocker, i.e. as though adding it were what unlocks the row.
//     That was an untested causal claim. C4a-3's reconnaissance had already found the
//     real obstacle is the GATHER NEST (two grid bases per 8-element group), not the
//     width label; C4a-4 confirmed it by construction -- the I32x4 value is three lines,
//     the leaf it forces is the row's actual cost. The value is REAL now because a real
//     row dispatches on it, which is the only thing this header ever certifies.
//
// The rule those two corrections exist to serve: this header never advertises a
// capability the emitter lacks. Note WHERE both wrong predictions lived -- not in the
// enums (the rule held there: neither value was ever declared early) but in the COMMENTS
// ABOUT the enums. The speculation simply moved to where the rule was not looking. Hence
// the standing instruction: a causal claim in this file is only as good as the ggml
// citation or the landed row next to it.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_GRIDDECODEPLAN_H
#define WEFT_SUPPORT_GRIDDECODEPLAN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace weft {

/// The grid codebook's per-entry packing width. NOT a cosmetic label: it fixes the
/// gather's byte-offset shift (idx*8 vs idx*4), the entry->grid-byte reinterpret, AND
/// -- the load-bearing one -- how many grid entries one 8-element activation group
/// consumes (see gridEntriesPerGroup). Because of that last one it is a LEAF-SELECTION
/// key in both repack emitters, not a parameter of a shared leaf.
///
/// WHAT THE I32x4 ROW ACTUALLY COST (C4a-4, landing iq3_xxs). This enum spent three
/// rounds as a ONE-value enum while the file's prose asserted iq3_xxs "needs
/// GridEntryWidth::I32x4" -- phrasing that reads as "the missing enum value is the
/// blocker". Landing the row falsified that. The width-as-parameter reading (shift
/// 3 -> 2, inner byte loop 0..7 -> 0..3) is real but trivial; the actual work is
/// STRUCTURAL and the width is only its PROXY:
///
///   * ENTRIES PER 8-ELEMENT GROUP goes 1 -> 2. Every I64x8 row consumes exactly ONE
///     grid entry per group of 8 activations (ggml iq1_s: `grid = iq1s_grid + idx;
///     for j<8: q8[j]*grid[j]`). iq3_xxs consumes TWO, because a 4-byte entry only
///     covers half a group (ggml ggml_vec_dot_iq3_xxs_q8_K):
///         grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
///         grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
///         for (j = 0; j < 4; ++j) {
///           sumi += grid1[j] * q8[j+0] * sign(j+0);
///           sumi += grid2[j] * q8[j+4] * sign(j+4);
///         }
///     Cross-checkable from the layout alone: block_iq3_xxs.qs is 3*QK_K/8 = 96 B,
///     split at QK_K/4 = 64 (`gas = x[i].qs + QK_K/4`) into a 64-byte index region +
///     8 uint32 aux words -- 8 indices per sub-block over 4 groups = 2 per group,
///     matching ggml's own `q3 += 8` per sub-block.
///   * CONSEQUENCE, and the reason this is a leaf: the innermost nest of every I64x8
///     leaf hoists ONE gridBase per group and walks j = 0..7 against it. The I32x4
///     nest hoists TWO bases and SPLITS the activation range -- lanes 0..3 read
///     gridBase1[j], lanes 4..7 read gridBase2[j-4]. Forcing that into the j<8
///     single-base shape does not fail loudly; it silently reads the WRONG grid bytes
///     for half of every group. The repacked grid-index strip doubles with it
///     (gidx[8][4][16] = 512 u8 -> gidx[8][8][16] = 1024 u8).
///
/// The SIGN plane, by contrast, is REUSED wholesale: `ksigns_iq2xs[(aux32 >> 7*l) & 127]`
/// tested by `kmask_iq2xs[j]` is EXACTLY iq2_xxs's mechanism, the selector is still ONE
/// per 8-element group (kmask bits 0..3 for the grid1 lanes, 4..7 for the grid2 lanes),
/// and GridSignPlane::Signs64 already is that plane. See GridSignPlane::Signs64.
enum class GridEntryWidth {
  I64x8, ///< 8 packed int8 per uint64 entry (iq2_xxs / iq2_xs / iq2_s / iq1_s /
         ///< iq1_m). ggml: uint64_t iq2xxs_grid[256] / iq2xs_grid[512] /
         ///< iq2s_grid[1024] / iq1s_grid[2048]. ONE entry per 8-element group.
  I32x4, ///< 4 packed int8 per uint32 entry (iq3_xxs). ggml: uint32_t
         ///< iq3xxs_grid[256]. TWO entries per 8-element group -- the structural
         ///< half of this axis; see the doc above.
};

/// How many grid entries one 8-element activation group consumes. DERIVED from the
/// entry width (8 grid bytes are needed per group; an entry supplies its width), NOT an
/// independent axis -- a row cannot pick these separately, and pretending otherwise
/// would invite a plan whose two fields disagree.
constexpr int gridEntriesPerGroup(GridEntryWidth w) {
  return w == GridEntryWidth::I64x8 ? 1 : 2;
}

/// The byte-offset shift the gather applies to a grid index (index -> index*bytes).
/// DERIVED from the entry width for the same reason.
constexpr int gridEntryByteShift(GridEntryWidth w) {
  return w == GridEntryWidth::I64x8 ? 3 : 2;
}

/// The sign plane the decode folds onto the gathered grid bytes. The Signs64 /
/// Signs256 rows differ ONLY in the DERIVED table's contents -- which is exactly
/// why the plan can name the table as data instead of the emitter branching on a
/// variant enum. TernaryDelta is NOT a table at all: it is the ABSENCE of a sign
/// plane (the iq1_s grid bytes are ALREADY signed ternary), so a TernaryDelta row
/// carries an EMPTY signArrayName and its weight sign-plane byte offset addresses
/// the per-sub-block +-1 DELTA strip instead of a sign SELECTOR strip.
enum class GridSignPlane {
  /// 7-bit selector -> the DERIVED +-1 plane built from ggml's ksigns_iq2xs[128]
  /// (kIQ2XXSKsigns), which the emitter expands to weft_iq2xxs_signs64[1024] = 128
  /// selectors x 8 +-1 bytes (iq2_xxs, iq2_xs). The "64" is the 64-BIT sign word a
  /// selector yields, NOT an entry count -- C4a-3 corrected this doc, which
  /// previously read "64-entry" and had misled the C4a/C4a-2 comments into
  /// forward-declaring a phantom Ksigns128 for iq3_xxs (whose sign plane is this
  /// SAME ksigns_iq2xs mechanism; see GridEntryWidth).
  Signs64,
  Signs256, ///< DIRECT 256-entry plane (iq2_s).
  /// NO sign plane; the grid bytes are ALREADY signed ternary, so the gathered byte
  /// IS the weight, and the sign-plane byte-offset slot carries a +-1 DELTA strip
  /// instead: per SUB-BLOCK for iq1_s, per 8-element GROUP for iq1_m.
  TernaryDelta,
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
///   SignScaleStore: sumf += cvt(sumi) * (d_x*d_y); a single constant is applied ONCE
///     at the STORE, and there is a SINGLE integer accumulator (iq2_xxs/xs/s, iq3_xxs).
///     The constant itself is NOT part of the shape -- it rides the plan's
///     storeScaleLiteral (0.125f for the iq2 rows, 0.25f for iq3_xxs). C4a-4 renamed
///     this value from `SignScaleEighth` for exactly that reason: ggml's iq3_xxs stores
///     `*s = 0.25f * sumf` while every iq2 row stores `*s = 0.125f * sumf`, so a value
///     literally named for one of the two constants would have had to carry the other
///     -- a name/reality mismatch, i.e. the next false claim in a file that has already
///     shipped two. The FOLD SHAPE is shared and really is shared; the CONSTANT is data.
///   DeltaGrid: sumf += (d_x*d_y) * (cvt(sumi) + 0.125f*cvt(sumi1)); TWO integer
///     accumulators (the grid dot AND the delta-bsum term), the 0.125 is applied
///     per-block to sumi1 ONLY, and there is NO trailing store-side factor. The
///     sumi1 term consumes the ACTIVATION bsums, which the iq2 rows never read
///     (iq1_s).
///   DeltaGridGroupSum: the SAME end-of-block float expression as DeltaGrid, but a
///     STRUCTURALLY different delta term, which is why it cannot ride the DeltaGrid
///     leaf (iq1_m). Two facts force the split, both checkable against ggml's
///     ggml_vec_dot_iq1_m_q8_K:
///       (1) delta granularity. iq1_s carries ONE +-1 per 32-element sub-block (qh
///           bit15). iq1_m carries FOUR -- one per 8-element GROUP (qh[l/2] bits
///           0x08/0x80) -- and they are INDEPENDENT bits.
///       (2) the delta term's activation factor. iq1_s sums 32 activations, which is
///           exactly bsums[2ib]+bsums[2ib+1], so it reads the bsums plane. iq1_m needs
///           the per-GROUP-of-8 activation sum, and block_q8_K's bsums are sums over
///           groups of SIXTEEN (ggml-common.h: `int16_t bsums[QK_K/16]`). The two
///           8-groups inside one bsums group carry INDEPENDENT delta signs, so their
///           sum CANNOT be recovered from a bsums entry: the bsums plane is not merely
///           unused by iq1_m, it is INEXPRESSIVE for it. The group-of-8 sum is instead
///           accumulated IN-KERNEL from the same 8 activation scalars the grid dot
///           already reads, and an iq1_m row therefore stamps NO
///           activation_bsums_byte_offset at all.
///     (iq1_m also folds DUAL ls, but that is the existing GridLsArity::Dual axis --
///     REUSED from iq2_xs/iq2_s, not a reason for this value to exist.)
enum class GridFoldArith {
  /// single i32 acc, trailing storeScaleLiteral at the store (iq2_xxs/xs/s, iq3_xxs).
  SignScaleStore,
  DeltaGrid,       ///< sumi + 0.125*sumi1 dual-acc bsums delta fold (iq1_s).
  /// sumi1 + 0.125*sumi2 dual-acc, per-GROUP delta + IN-KERNEL group-of-8
  /// activation sums (bsums are per-16 => inexpressive here) (iq1_m).
  DeltaGridGroupSum,
};

/// One registered grid decode model. The plan is pure DATA: it NAMES the DERIVED
/// emit-period tables (never their literals) and pins the decode axes.
///
/// LEAF SELECTION, in the order both repack emitters MUST apply it -- the order is a
/// correctness rule, not a style preference, because each key is a SUBSET trap for the
/// one after it:
///   1. foldArith == DeltaGrid          -> the iq1_s leaf.
///   2. foldArith == DeltaGridGroupSum  -> the iq1_m leaf. (iq1_m's lsArity IS Dual, so
///      deciding on ls arity first would silently hand it to the iq2 dual-ls leaf --
///      C4a-3's finding.)
///   3. entryWidth == I32x4             -> the iq3_xxs DUAL-ENTRY leaf. (iq3_xxs's
///      lsArity IS Single and its foldArith IS SignScaleStore -- it matches the iq2_xxs
///      leaf on EVERY key except this one, so deciding on ls arity first would silently
///      hand it to a leaf that hoists one grid base per group and shifts indices by 3.
///      C4a-4's finding: the SAME subset trap as iq1_m's, one axis over.)
///   4. lsArity == Single               -> the iq2_xxs leaf; otherwise the iq2 dual-ls
///      leaf.
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
  /// The C float literal the SignScaleStore fold multiplies in at the STORE, e.g.
  /// "0.125f" (iq2_xxs/xs/s) or "0.25f" (iq3_xxs). Carried as the LITERAL TEXT the
  /// emitter emits, not a float: the emitter's contract is a byte-exact C token, and
  /// round-tripping through a host float + formatter would put a printf between the
  /// registry and the emitted byte for no gain.
  /// EMPTY iff foldArith != SignScaleStore -- the DeltaGrid / DeltaGridGroupSum folds
  /// have NO store-side factor at all (their 0.125 is a per-block factor on the SECOND
  /// accumulator, which is their leaf's own constant, not this slot's).
  llvm::StringRef storeScaleLiteral;
};

/// The CLOSED registry lookup -- the single fail-closed authority the grid core
/// VERIFIER and both grid EMITTER leaves consult. Returns nullptr for an
/// unregistered decode_model, which every caller MUST treat as REJECT ([D-1]).
const GridDecodePlan *lookupGridDecodePlan(llvm::StringRef decodeModel);

/// The role marker the SignScaleStore leaves stamp on the store-side scale call. This
/// text SHIPS -- it reaches the emitted C as `callee=<role>` on the step comment -- so
/// it is DERIVED from storeScaleLiteral rather than carried as a second field: two
/// fields could drift, and the drifted state here is an emitted comment that NAMES A
/// CONSTANT IT IS NOT ATTACHED TO. That precise failure (a shipped annotation asserting
/// something untrue, waved through as cosmetic) is the one this family has already made
/// twice, so the mapping is CLOSED and returns an EMPTY marker for an unrecognized
/// literal -- the leaf then REFUSES to emit rather than mislabeling the store.
llvm::StringRef gridStoreScaleRole(llvm::StringRef storeScaleLiteral);

/// The registered plans, in registration order. Used to RENDER the verifier's
/// diagnostic so the accepted set can never drift from the registry.
llvm::ArrayRef<GridDecodePlan> getGridDecodePlans();

} // namespace weft

#endif // WEFT_SUPPORT_GRIDDECODEPLAN_H
