//===- GridDecodePlan.cpp - Grid-codebook repack decode plan registry -----===//
//
// See GridDecodePlan.h for the C4a [B-4] parameterization rationale.
//
//===----------------------------------------------------------------------===//

#include "Weft/Support/GridDecodePlan.h"

#include "llvm/ADT/StringSwitch.h"

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
/// promoting GridFoldArith::DeltaGridGroupSum. C4a-4 landed iq3_xxs the same four ways,
/// promoting GridEntryWidth::I32x4. C4a-5 landed iq3_s the same four ways and promoted
/// NOTHING -- the first row whose every axis value already existed. That is the whole
/// point of the exercise arriving, not a shortcut taken: the row still had to land all
/// four pieces, it just needed no new vocabulary to say what it is.
///
/// This comment previously said iq3_xxs/iq3_s "remain NOT registered ... GridEntryWidth::
/// I32x4 stays ABSENT from the enum until a row + front door + leaf + oracle land
/// together". The PROCESS half of that was right and C4a-4 honored it (all four landed
/// together). The CAUSAL half was a guess that read as fact -- it framed the absent enum
/// value as the thing standing between the registry and an iq3_xxs row. It was not. The
/// value is three lines; what actually had to be built is a new emitter leaf, because a
/// 4-byte grid entry covers only HALF an 8-element group and every existing leaf's
/// innermost nest is built around one entry covering a whole one. See GridDecodePlan.h's
/// GridEntryWidth doc. (C4a-3 had already retired this comment's OTHER prediction for
/// these rows -- a GridSignPlane::Ksigns128 that ggml shows is not needed at all.)
/// Two wrong predictions about one row, both stated in comments rather than enums, is
/// the reason the surviving text cites ggml or a landed row for every causal claim.
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
        /*foldArith=*/GridFoldArith::SignScaleStore,
        /*storeScaleLiteral=*/"0.125f",
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
        /*foldArith=*/GridFoldArith::SignScaleStore,
        /*storeScaleLiteral=*/"0.125f",
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
        /*foldArith=*/GridFoldArith::SignScaleStore,
        /*storeScaleLiteral=*/"0.125f",
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
        // DeltaGrid has NO store-side factor (its 0.125 rides sumi1 per-block).
        /*storeScaleLiteral=*/"",
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
        // DeltaGridGroupSum has NO store-side factor either.
        /*storeScaleLiteral=*/"",
    },
    // iq3_xxs (C4a-4) -- the FIRST I32x4 row, and the one that shows the axes' limit:
    // it REUSES four of five axis values (Signs64, Single, SignScaleStore, and the
    // 256-entry count iq2_xxs also has) yet still needs a leaf of its own, because the
    // ONE value it does not reuse is the structural one.
    //
    // Against ggml_vec_dot_iq3_xxs_q8_K, what it SHARES with iq2_xxs is nearly the whole
    // decode: the same `ls = 2*(aux32 >> 28) + 1` single per-sub-block scale, the same
    // `ksigns_iq2xs[(aux32 >> 7*l) & 127]` 7-bit sign selector tested by `kmask_iq2xs[j]`
    // (hence Signs64, REUSED -- the DERIVED plane is byte-identical, so this row names
    // the SAME weft_iq2xxs_signs64 table iq2_xxs does rather than cloning it), the same
    // single-i32-accumulator `bsum += sumi*ls` / `sumf += d*bsum` chain, and the same
    // 256-entry grid size.
    //
    // What it does NOT share, and why this is a leaf:
    //   (a) ENTRY WIDTH. iq3xxs_grid is `uint32_t[256]` (ggml-common.h), not uint64_t.
    //       So one entry supplies 4 grid bytes, an 8-element group needs TWO of them
    //       (`grid1 = iq3xxs_grid + q3[2*l+0]`, `grid2 = iq3xxs_grid + q3[2*l+1]`, then
    //       `for j<4: grid1[j]*q8[j+0]; grid2[j]*q8[j+4]`), and the gather shift is 2
    //       not 3. The emitters key the dual-entry leaf off entryWidth == I32x4 for
    //       exactly this -- see GridDecodePlan.h's leaf-selection order, where this row
    //       is the reason step 3 must precede step 4.
    //   (b) STORE CONSTANT. ggml ends `*s = 0.25f * sumf`, not 0.125f. Same fold SHAPE,
    //       different constant -- which is now data (storeScaleLiteral) rather than a
    //       second enum value named for a number.
    //
    // Its repack layout is this line's design and follows iq2_xxs's exactly except that
    // the grid-index strip doubles: ggml's block_iq3_xxs is 98 B (fp16 d + qs[3*QK_K/8]
    // = 96 B), where the qs region SPLITS at QK_K/4 = 64 (`gas = x[i].qs + QK_K/4`) into
    // 64 raw u8 grid indices + 8 uint32 aux words, each aux word packing one sub-block's
    // ls (bits 28-31) and its four 7-bit sign selectors (bits 0-27). The repack pulls
    // those apart ONCE into flat strides so the kernel never touches an aux word:
    //   d[16]           fp16 @ +0    (32 B)
    //   ls[8][16]       int8 @ +32   (128 B)  2*(aux32>>28)+1, in [1,31]
    //   gidx[8][8][16]  u8   @ +160  (1024 B) the raw q3 bytes, TWO per group
    //   ssel[8][4][16]  u8   @ +1184 (512 B)  (aux32 >> 7*l) & 127, ONE per group
    //   ------------------------------------- stride 1696
    {
        /*decodeModel=*/"iq3_xxs",
        // REUSES iq2_xxs's DERIVED sign plane BY NAME: ggml's iq3_xxs sign mechanism is
        // ksigns_iq2xs, the same table, so emitting a second identical 1024-byte array
        // under an iq3 name would be a copy, not a fact.
        /*gridArrayName=*/"weft_iq3xxs_grid",
        /*signArrayName=*/"weft_iq2xxs_signs64",
        /*gridEntryCount=*/256,
        /*entryWidth=*/GridEntryWidth::I32x4,
        /*signPlane=*/GridSignPlane::Signs64,
        /*lsArity=*/GridLsArity::Single,
        /*foldArith=*/GridFoldArith::SignScaleStore,
        /*storeScaleLiteral=*/"0.25f", // ggml: *s = 0.25f * sumf (NOT 0.125f).
    },
    // iq3_s (C4a-5) -- the LAST grid sibling, and the only one that cost the enums
    // NOTHING. Every axis value below already existed, put there by a DIFFERENT row:
    //   * I32x4          from iq3_xxs (C4a-4) -- and with it that row's dual-entry LEAF,
    //                    reused unchanged in shape.
    //   * Signs256       from iq2_s.
    //   * 512 entries    iq2_xs's count.
    //   * Single         iq2_xxs's / iq3_xxs's.
    //   * SignScaleStore iq2_xxs's / iq3_xxs's.
    // Read off ggml_vec_dot_iq3_s_q8_K, axis by axis, rather than assumed from the family
    // resemblance -- because on two of these axes the resemblance points the wrong way:
    //
    //   (a) SIGN PLANE = Signs256, NOT iq3_xxs's Signs64. This is the axis where iq3_s
    //       breaks with its own xxs sibling and sides with iq2_s. iq3_xxs derives its
    //       signs INDIRECTLY -- `ksigns_iq2xs[(aux32 >> 7*l) & 127]` unpacks a 7-bit
    //       SELECTOR out of an aux word into an 8-bit mask. iq3_s has no aux word and no
    //       selector: block_iq3_s carries an EXPLICIT `uint8_t signs[QK_K/8]` plane, and
    //       the decode tests its raw byte directly (`signs[l] & kmask_iq2xs[j]`), which is
    //       verbatim what ggml_vec_dot_iq2_s_q8_K does. So the DERIVED +-1 plane must be
    //       indexed by all 256 byte values, not 128 selectors, and this row names iq2_s's
    //       weft_iq2s_signs256 -- the SAME reuse-don't-clone judgment iq3_xxs made for
    //       signs64, and legitimate for the same reason: the plane is a pure function of
    //       (byte, j), so a second 2048-byte decl under an iq3 name would be a copy, not
    //       a fact about iq3_s. NOTE the gather ARITHMETIC is identical either way
    //       (`table[b*8 + j]`, shift 3) -- Signs64 vs Signs256 differ only in the table
    //       NAMED and its size, which is exactly why the plan can carry this as data and
    //       the leaf needs no branch on it.
    //   (b) LS ARITY = Single, NOT Dual. ggml's loop reads `ls1`/`ls2` from the two
    //       nibbles of `scales[ib32/2]` -- but it steps `ib32 += 2` and spends ls1 on all
    //       4 groups of sub-block ib32 and ls2 on all 4 of ib32+1. One ls per 32-element
    //       sub-block = Single; the byte just holds two sub-blocks' worth. (C4a-4's [D-1]
    //       probe predicted Dual here. See GridLsArity.)
    //
    // Its 9-bit grid index (`qs[e] | (((qh[ib] >> e) & 1) << 8)`, 0..511) is assembled at
    // REPACK time into a u16 strip, as iq2_xs's 9-bit and iq1_s's 11-bit indices already
    // are; the leaf reads the strip width from gridIndexStripIsU16(gridEntryCount).
    //
    // Repack layout (this line's design, iq3_xxs's with the index strip widened to u16):
    //   d[16]            fp16 @ +0     (32 B)
    //   ls[8][16]        int8 @ +32    (128 B)  2*nibble+1, in [1,31]
    //   gidx[8][8][16]   u16  @ +160   (2048 B) the 9-bit index, TWO per group
    //   signs[8][4][16]  u8   @ +2208  (512 B)  the EXPLICIT sign byte, ONE per group
    //   -------------------------------------- stride 2720
    {
        /*decodeModel=*/"iq3_s",
        /*gridArrayName=*/"weft_iq3s_grid",
        // REUSES iq2_s's DERIVED sign plane BY NAME -- see (a) above.
        /*signArrayName=*/"weft_iq2s_signs256",
        /*gridEntryCount=*/512,
        /*entryWidth=*/GridEntryWidth::I32x4,
        /*signPlane=*/GridSignPlane::Signs256,
        /*lsArity=*/GridLsArity::Single,
        /*foldArith=*/GridFoldArith::SignScaleStore,
        // ggml: *s = sumf -- NO store constant. Carried as the literal 1.0f the emitter
        // emits; see GridDecodePlan::storeScaleLiteral for why this is written as a
        // multiply by one rather than as an empty field.
        /*storeScaleLiteral=*/"1.0f",
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

llvm::StringRef weft::gridStoreScaleRole(llvm::StringRef storeScaleLiteral) {
  // CLOSED by construction: every SignScaleStore row's literal must appear here or its
  // leaf refuses to emit. "eighth_scale" is the marker the iq2 rows have always shipped
  // and is kept EXACTLY so their emitted C stays byte-identical; "quarter_scale" is
  // iq3_xxs's, and it exists because reusing "eighth_scale" for ggml's 0.25f store would
  // have shipped a comment that names the wrong constant. "unit_scale" is iq3_s's, whose
  // ggml store `*s = sumf` applies the constant 1.
  //
  // The EMPTY literal deliberately has NO case and therefore NO marker: that is what makes
  // "a SignScaleStore row forgot its storeScaleLiteral" a REFUSAL rather than an unscaled
  // store. iq3_s is the row that tested whether that gate would survive contact with a
  // format that genuinely has no store constant -- it did, because the constant is written
  // as 1.0f rather than as absence. See GridDecodePlan::storeScaleLiteral.
  return llvm::StringSwitch<llvm::StringRef>(storeScaleLiteral)
      .Case("0.125f", "eighth_scale")
      .Case("0.25f", "quarter_scale")
      .Case("1.0f", "unit_scale")
      .Default("");
}
