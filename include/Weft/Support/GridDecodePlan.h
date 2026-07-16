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
// iq1_m). The enums below carry ONLY axis values those rows exercise. C4a-2 landed
// iq1_s FOUR-ways-at-once (registry row + front door + emitter leaves + byte-exact
// oracle), which is what promoted GridSignPlane::TernaryDelta and
// GridFoldArith::DeltaGrid from "specified in the C4a report" to real axis values
// with a real row using them -- C4a deliberately refused to forward-declare them
// while no row existed. C4a-3 landed iq1_m the same four ways, promoting
// GridFoldArith::DeltaGridGroupSum the same way; iq1_m REUSED three existing axis
// values unchanged (I64x8, TernaryDelta, and -- from the iq2 dual rows -- Dual), so
// the marginal enum cost of the 5th row was ONE value.
//
// The still-unregistered iq3_xxs / iq3_s need GridEntryWidth::I32x4, which remains
// ABSENT for the SAME reason: this header never advertises a capability the emitter
// lacks. GridEntryWidth is still a ONE-value enum, and C4a-3's iq3_xxs
// reconnaissance is why it stayed that way -- see the GridEntryWidth doc below for
// what an I32x4 row would actually cost, which is NOT a shift-amount parameter.
//
// C4a-3 ALSO retired a forward-declaration this header used to make: the C4a/C4a-2
// text listed GridSignPlane::Ksigns128 as a second axis value iq3_xxs "needs". It
// does not -- iq3_xxs's sign plane is the ksigns_iq2xs mechanism GridSignPlane::
// Signs64 already carries. That claim survived two rounds only because nobody
// checked it against ggml, which is precisely the failure mode "no enum value
// without a real row using it" exists to prevent: the rule was written for the
// ENUM, but the SPECULATION had simply moved into the comments instead.
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
///
/// WHY THIS ENUM STILL HAS ONE VALUE (C4a-3 iq3_xxs reconnaissance). The obvious
/// reading of "I32x4" is that it is a PARAMETER of the existing leaves -- the
/// gather byte-offset shift would go 3 -> 2 (idx*8 -> idx*4) and the inner byte
/// loop 0..7 -> 0..3. Both of those really are parameters. But they are not the
/// work, and the honest finding of C4a-3's reconnaissance is that the entry width
/// is a PROXY for a change that is structural:
///
///   * ENTRIES PER 8-ELEMENT GROUP goes 1 -> 2, and that is the non-translation.
///     Every registered row consumes exactly ONE grid entry per group of 8
///     activations (ggml iq1_s: `grid = iq1s_grid + idx; for j<8: q8[j]*grid[j]`).
///     iq3_xxs consumes TWO (`grid1 = iq3xxs_grid + q3[2l+0]`,
///     `grid2 = iq3xxs_grid + q3[2l+1]`, then `for j<4: grid1[j]*q8[j+0] ...
///     grid2[j]*q8[j+4]`) -- because a 4-byte entry only covers half a group.
///     Checkable: block_iq3_xxs.qs is 3*QK_K/8 = 96 B split at QK_K/4 = 64 into a
///     64-byte index region + 8 uint32 aux words, i.e. 8 indices per sub-block
///     over 4 groups = 2 per group (and ggml's own `q3 += 8` per sub-block).
///     Consequences: the repacked grid-index strip DOUBLES (gidx[8][4][16] = 512
///     u16 per column-group -> gidx[8][8][16] = 1024), and the emitter's innermost
///     nest -- which today hoists ONE gridBase per group and then walks j = 0..7
///     against it -- has to become two bases with a split activation range. That is
///     a new leaf, not a new constant.
///   * The trailing fold constant is 0.25f, not 0.125f, and it rides the STORE
///     (ggml: `*s = 0.25f * sumf`). The shape is SignScaleEighth's, but that value
///     is literally named for its constant, so it would need parameterizing or
///     splitting.
///
/// And, symmetrically, the C4a/C4a-2 comments' claim that iq3_xxs needs a NEW
/// GridSignPlane::Ksigns128 is SPECULATION THAT THE CODE CONTRADICTS -- C4a-3
/// checked it rather than inheriting it. iq3_xxs's sign plane is
/// `ksigns_iq2xs[(aux32 >> 7l) & 127]` tested by `kmask_iq2xs[j]`, which is EXACTLY
/// iq2_xxs's mechanism (`ksigns_iq2xs[(aux32[1] >> 7l) & 127]` / `kmask_iq2xs[j]`),
/// and GridSignPlane::Signs64 ALREADY IS that plane: kIQ2XXSKsigns is
/// `std::array<std::int32_t, 128>` (ggml's ksigns_iq2xs verbatim) and the emitter
/// expands it to `weft_iq2xxs_signs64[1024]` = 128 selectors x 8 +-1 bytes. So an
/// iq3_xxs row would REUSE Signs64. (The "64" in that value's name is the 64-BIT
/// sign word each selector yields, not an entry count; the doc below said
/// "64-entry", which was wrong, and C4a-3 corrected it.) Ksigns128 is therefore NOT
/// listed as a pending axis value anywhere in this header -- an enum value nobody
/// needs is exactly the forward-declaration this file refuses to make.
enum class GridEntryWidth {
  I64x8, ///< 8 packed int8 per uint64 entry (iq2_xxs / iq2_xs / iq2_s / iq1_s /
         ///< iq1_m). ggml: uint64_t iq2xxs_grid[256] / iq2xs_grid[512] /
         ///< iq2s_grid[1024] / iq1s_grid[2048].
};

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
///   SignScaleEighth: sumf += cvt(sumi) * (d_x*d_y); the 0.125 is applied ONCE at
///     the STORE, and there is a SINGLE integer accumulator (iq2_xxs/xs/s).
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
  SignScaleEighth, ///< single i32 acc, trailing 0.125 at store (iq2_xxs/xs/s).
  DeltaGrid,       ///< sumi + 0.125*sumi1 dual-acc bsums delta fold (iq1_s).
  /// sumi1 + 0.125*sumi2 dual-acc, per-GROUP delta + IN-KERNEL group-of-8
  /// activation sums (bsums are per-16 => inexpressive here) (iq1_m).
  DeltaGridGroupSum,
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
