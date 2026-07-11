#ifndef TIANCHENRV_DIALECT_IME_IR_IMEOPS_TD
#define TIANCHENRV_DIALECT_IME_IR_IMEOPS_TD

include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"

def TCRV_IME_Dialect : Dialect {
  let name = "tcrv_ime";
  let cppNamespace = "::tianchenrv::tcrv::ime";
  let summary = "TianChen-RV IME (Integer Matrix Extension) extension dialect";
  let description = [{
    The IME dialect is the plugin-local extension-family surface for the
    Spacemit X60 Integer Matrix Extension (IME1). The architectural family is
    tcrv.ime; the concrete MLIR dialect namespace is tcrv_ime because MLIR
    dialect namespaces cannot contain '.' characters.

    IME is a non-RVV, RISC-V vendor matrix extension that rides the SAME unified
    TCRV common pipeline RVV uses (capability-driven proposal / legality /
    selection / boundary materialization / common EmitC route). It is the N2
    second-family target: it proves a non-RVV family slots into the common path
    with zero core-family branches.

    This dialect exposes a real int8->int32 matrix-multiply-accumulate execution
    op (the `vmadot` MAC unit) plus its selected lowering-boundary metadata. It
    is an IME EXECUTION op, not a high-level matmul/tile/tensor op: it carries
    the structural executable facts (element-in dtype, accumulator dtype, the
    VLEN-derived MAC fragment shape) so emission is structural (core-invariants
    I5), not inferred from names.
  }];
}

class TCRVIME_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_IME_Dialect, mnemonic, traits>;

def MMAOp : TCRVIME_Op<"mma",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME int8->int32 matrix-multiply-accumulate (vmadot) boundary";
  let description = [{
    The selected IME execution boundary for a chosen tcrv.exec.variant: an
    int8->int32 matrix-multiply-accumulate C += A . B^T realized by the IME1
    `vmadot` MAC unit. A is (M,K) int8, B is stored (N,K) int8 (== B^T of the
    math (K,N) matrix), C is (M,N) int32. The MAC fragment shape (M,N,K) is a
    capability-derived fact (from VLEN/SEW): at VLEN=256/SEW=8 the X60 unit is
    4x4x8.

    The op carries the structural executable facts (elem_in_bits=8,
    accum_bits=32, mac_m/mac_n/mac_k, ime_op="vmadot") plus the selected-path
    provenance (source kernel, selected variant, origin plugin, role, status,
    required capabilities). A fail-closed verifier (core-invariants I7) admits
    ONLY the validated IME1 int8->int32 envelope; anything else is rejected, so
    no body outside the proven envelope is ever emitted.

    This is an IME execution op, not a high-level matmul op. It does not model
    generic tensor/tile compute, and it makes no correctness/performance claim
    in the IR itself; correctness is proven by the emitted kernel on real K1.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MMAUOp : TCRVIME_Op<"mma_u",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME uint8->int32 matrix-multiply-accumulate (vmadotu) boundary";
  let description = [{
    The SECOND IME execution boundary (plugin breadth, N2): the UNSIGNED
    int8 form of the MAC unit, realized by the IME1 `vmadotu` instruction.
    Identical fragment-shape envelope to tcrv.ime.mma (capability-derived from
    VLEN/SEW: at VLEN=256/SEW=8 the X60 unit is 4x4x8, int32 accumulator), but a
    GENUINELY DIFFERENT emitted instruction: `vmadotu` (unsigned*unsigned, IME1
    signedness variant) computes C += A . B^T treating A and B as UNSIGNED
    int8, vs tcrv.ime.mma's signed `vmadot`. The two have distinct encodings
    (vmadot=0xe210312b vs vmadotu=0xe210012b) and distinct numeric semantics; on
    the same all-high-bit-set int8 data the unsigned MAC differs from the signed
    MAC, so this is not a renamed op.

    Both ops gate on the SAME first-class `spacemit.ime` capability FACT (the
    `xsmtvdotii` march token assembles all four signedness forms — proven by the
    SpacemiT GCC15 assembler). Which signedness boundary materializes is a
    capability-derived fact (the `ime_signedness` request), NOT a family-name
    string and NOT a second capability id. A fail-closed verifier (I7) admits
    ONLY the validated IME1 uint8->int32 `vmadotu` envelope.

    This is an IME execution op, not a high-level matmul op. It makes no
    correctness/performance claim in the IR; correctness is proven by the emitted
    kernel bit-exact vs an unsigned scalar reference on real K1.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MMASUOp : TCRVIME_Op<"mma_su",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME mixed-sign int8->int32 matrix-multiply-accumulate (vmadotsu) boundary";
  let description = [{
    The FOURTH IME execution boundary (plugin breadth, N2 rapid-add): the
    MIXED-SIGN form of the MAC unit, realized by the IME1 `vmadotsu`
    instruction. This is the canonical quantized mixed-sign case: A is treated
    as SIGNED int8 (activations) and B as UNSIGNED int8 (weights), C += A . B^T
    accumulated in int32. Identical fragment-shape envelope to tcrv.ime.mma
    (capability-derived from VLEN/SEW: at VLEN=256/SEW=8 the X60 unit is 4x4x8,
    int32 accumulator), but a GENUINELY DIFFERENT emitted instruction:
    `vmadotsu` (signed*unsigned int8 MAC, IME1 mixed-sign signedness variant)
    computes a numerically different result than tcrv.ime.mma's `vmadot`
    (signed*signed) and tcrv.ime.mma_u's `vmadotu` (unsigned*unsigned). The
    three have distinct encodings (vmadot=0xe210312b, vmadotu=0xe210012b,
    vmadotsu=0xe210232b) and distinct numeric semantics; on the same
    all-high-bit-set int8 data the mixed-sign MAC differs from BOTH the signed
    and the unsigned MAC, so this is not a renamed op.

    All four ops gate on the SAME first-class `spacemit.ime` capability FACT (the
    `xsmtvdotii` march token assembles all four signedness forms — proven by the
    SpacemiT GCC15 assembler). Which signedness boundary materializes is a
    capability-derived fact (the `ime_signedness` request), NOT a family-name
    string and NOT a second capability id. A fail-closed verifier (I7) admits
    ONLY the validated IME1 mixed-sign int8->int32 `vmadotsu` envelope.

    This is an IME execution op, not a high-level matmul op. It makes no
    correctness/performance claim in the IR; correctness is proven by the emitted
    kernel bit-exact vs a mixed-sign scalar reference on real K1.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MMAUSOp : TCRVIME_Op<"mma_us",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME mixed-sign int8->int32 matrix-multiply-accumulate (vmadotus) boundary";
  let description = [{
    The SIXTH IME execution boundary (plugin breadth, N2 rapid-add): the OTHER
    MIXED-SIGN form of the MAC unit, realized by the IME1 `vmadotus`
    instruction — the reversed-order sibling of tcrv.ime.mma_su. Here A is
    treated as UNSIGNED int8 and B as SIGNED int8, C += A . B^T accumulated in
    int32. This completes the signedness family {ss=vmadot, uu=vmadotu,
    su=vmadotsu, us=vmadotus}. Identical fragment-shape envelope to tcrv.ime.mma
    (capability-derived from VLEN/SEW: at VLEN=256/SEW=8 the X60 unit is 4x4x8,
    int32 accumulator), but a GENUINELY DIFFERENT emitted instruction:
    `vmadotus` (unsigned A * signed B int8 MAC, IME1 mixed-sign signedness
    variant) computes a numerically different result than tcrv.ime.mma_su's
    `vmadotsu` (signed A * unsigned B), and differs from BOTH the pure-signed
    `vmadot` and the pure-unsigned `vmadotu`. The four signedness forms have
    distinct encodings (vmadot=0xe210312b, vmadotu=0xe210012b, vmadotsu=0xe210212b,
    vmadotus=0xe210112b) and distinct numeric semantics; on the same mixed
    high-bit int8 data the four MACs are pairwise distinct, so this is not a
    renamed op.

    All IME MAC ops gate on the SAME first-class `spacemit.ime` capability FACT
    (the `xsmtvdotii` march token assembles all four signedness forms — proven by
    the SpacemiT GCC15 assembler). Which signedness boundary materializes is a
    capability-derived fact (the `ime_signedness` request), NOT a family-name
    string and NOT a second capability id. A fail-closed verifier (I7) admits
    ONLY the validated IME1 mixed-sign int8->int32 `vmadotus` envelope.

    This is an IME execution op, not a high-level matmul op. It makes no
    correctness/performance claim in the IR; correctness is proven by the emitted
    kernel bit-exact vs a mixed-sign scalar reference on real K1.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MMASlideOp : TCRVIME_Op<"mma_slide",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME int8->int32 sliding-window MAC (vmadot1/2/3) boundary";
  let description = [{
    The FIFTH IME execution boundary (plugin breadth, N2 rapid-add): the
    SLIDING-WINDOW form of the MAC unit, realized by the IME1 `vmadot1`/`vmadot2`/
    `vmadot3` instructions (Xsmti8i32mm_slide, K1's SECOND IME1 sub-extension).
    This is a genuinely DIFFERENT kernel SHAPE, not a fifth signedness: A is read
    from an EVEN VS1:VS1+1 register PAIR holding a 2*M x K = 8x8 int8 A-block, and
    the `slide` field (1/2/3) shifts the A read-window DOWN by `slide` rows. So one
    loaded 8x8 A-pair feeds overlapping 4x8 matmuls (the 1-D sliding-window /
    strided-A-reuse convolution primitive): vmadot reads A rows 0..3, vmadot1 rows
    1..4, vmadot2 rows 2..5, vmadot3 rows 3..6. B (VS2) and C (even VD pair, 4x4
    int32) are identical to the non-slide MAC. funct7 is `111001` (e6...), DISTINCT
    from the non-slide `111000` (e2...): vmadot1 v4,v0,v2 = 0xe620322b vs
    vmadot v4,v0,v2 = 0xe220322b. The slide semantics were CONFIRMED bit-exact on
    real K1 (X60) by a 4-way window discriminator (vmadot==rows0..3 anchor,
    vmadot1==rows1..4, != all other windows).

    All IME ops gate on the SAME first-class `spacemit.ime` capability FACT (the
    `xsmtvdotii` march token assembles BOTH the non-slide and the slide families).
    Which boundary materializes is a capability-derived fact (the non-slide
    `ime_signedness` request vs this op's `ime_slide` request), NOT a family-name
    string and NOT a second capability id. A fail-closed verifier (I7) admits ONLY
    the validated IME1 slide int8->int32 envelope with `slide` in {1,2,3}.

    This is an IME execution op, not a high-level matmul op. It makes no
    correctness/performance claim in the IR; correctness is proven by the emitted
    kernel bit-exact vs a slide scalar reference on real K1.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    I64Attr:$slide,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MatMulOp : TCRVIME_Op<"matmul",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME int8->int32 tiled whole-matrix multiply-accumulate boundary";
  let description = [{
    The selected IME execution boundary for a WHOLE-MATRIX int8->int32
    multiply-accumulate C[M,N] += A[M,K] . B[K,N] realized by tiling the IME1
    MAC fragment (`vmadot` / `vmadotu`) over the (M/mac_m) x (N/mac_n) x
    (K/mac_k) tile grid. This is the same capability-derived MAC fragment
    (4x4x8 at VLEN=256/SEW=8) as tcrv.ime.mma, but applied to a problem-sized
    matmul rather than a single fragment — so it carries the problem dims
    (mat_m/mat_n/mat_k) IN ADDITION to the fragment shape (mac_m/mac_n/mac_k).

    A is stored (M,K) int8 row-major; B is stored (N,K) int8 row-major (== B^T
    of the math (K,N) matrix — the IME-native B^T layout); C is (M,N) int32
    row-major. The kernel keeps each 4x4 C output tile in the int32 VD pair
    across the WHOLE K reduction (the MAC accumulates in-register), storing C
    once per tile — so the per-output reduction the RVV path needs (vredsum) is
    structurally absent. The signedness form (`vmadot` signed / `vmadotu`
    unsigned) follows the same capability-derived fact as tcrv.ime.mma/mma_u.

    A fail-closed verifier (core-invariants I7) admits ONLY the validated IME1
    int8->int32 envelope AND requires the problem dims to be a whole multiple of
    the MAC fragment (no remainder path — honest and bounded). This is an IME
    execution op, not a high-level matmul op: it carries the structural
    executable facts so emission is structural (I5); it makes no
    correctness/performance claim in the IR itself.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    I64Attr:$mat_m,
    I64Attr:$mat_n,
    I64Attr:$mat_k,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// G4 M1a: the FORMAT-KEYED q4_0 IME GEMM tile typed-region front door.
//
// The ops above (mma / matmul / ...) are FORMAT-AGNOSTIC: their inputs are
// already int8 (the tiled tcrv.ime.matmul consumes pre-packed int8 A/B). The
// three ops below add the FIRST format-keyed IME tile: a q4_0-quantized weight
// matrix decoded (offset-binary nibble) into the int8 MAC fragment, fed to the
// int8->int32 vmadot MAC, in-register K-accumulated. This mirrors the RVV
// typed_repack front door (RVVLowerQuantContraction lowerToRepackGemm) but
// targets the IME MATRIX paradigm (vmadot MAC) instead of the RVV vector
// paradigm (block-dot). Correctness of the int8->int32 MAC is int32-EXACT (no
// float tolerance); the vmadot instruction itself is proven bit-exact on real
// K1 (M1b board seal). This front door is CONSTRUCTED (typed region), landed
// but BOARD-PENDING: the coverage cell gemm_tile/q4_0@ime stays blocked-on-IME
// until the K1 silicon seal.
//===----------------------------------------------------------------------===//

def Q40DequantCoreOp : TCRVIME_Op<"q4_0_dequant_core"> {
  let summary = "IME q4_0 -> int8 offset-binary nibble dequant core brick";
  let description = [{
    The typed q4_0 WEIGHT DECODE primitive of the format-keyed IME GEMM tile
    (the format-keyed sibling of the RVV repack decode brick). It takes the
    per-tile contraction-block induction variable `block_index` and produces the
    decoded 4x8 int8 B-fragment (a builtin vector<32xi8>): the ggml q4_0 block is
    (fp16 d, uint8 qs[16]) = 18 bytes carrying 32 packed nibbles; each 4-bit
    nibble q in [0,15] decodes to the OFFSET-BINARY int8 quant value (q - 8) in
    [-8,7], which fits int8 exactly, so the decoded fragment feeds the int8->int32
    vmadot MAC with NO precision loss. The per-block fp16 scale `d` is NOT applied
    here (it is a separate float fold AFTER the int32 MAC), so the decode is a
    PURE integer transform and the produced fragment is int32-MAC-exact.

    The op carries the bounded q4_0 layout FACTS (decode_model, qk=32, the 18-byte
    weight_block_stride, the +2 weight_quant_byte_offset past the fp16 d, and the
    +0 weight_scale_byte_offset) so emission is structural (I5): the emitter reads
    the op identity + these facts, never a q4/q8 name. For the bounded surface
    decode_model is limited to "q4_0_offset_binary_nibble".
  }];
  let arguments = (ins
    AnyType:$block_index,
    StrAttr:$decode_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$weight_scale_byte_offset
  );
  let results = (outs AnyType:$b_fragment);
  let assemblyFormat =
      "$block_index attr-dict `:` type($block_index) `->` type($b_fragment)";
  let hasVerifier = 1;
}

def VmadotMacLeafOp : TCRVIME_Op<"vmadot_mac_leaf"> {
  let summary = "IME int8->int32 vmadot MAC leaf brick (typed, not opaque asm)";
  let description = [{
    The typed vmadot MAC LEAF of the format-keyed IME GEMM tile: it consumes the
    4x8 int8 activation fragment `a_fragment` (vector<32xi8>), the decoded 4x8 int8
    weight fragment `b_fragment` (vector<32xi8>, the q4_0_dequant_core result), the
    contraction-block induction variable `block_index`, and the carried-IN 4x4
    int32 accumulator tile `acc_in` (vector<16xi32>), and produces the carried-OUT
    4x4 int32 accumulator tile `acc_out`: acc_out = acc_in + A . B^T computed by
    the IME1 `vmadot` MAC unit (int8*int8 -> int32, the FOUNDATION-validated 4x4x8
    fragment at VLEN=256/SEW=8). The MAC accumulates in int32 with NO overflow in
    the validated envelope (|q4_0 quant| <= 8, |q8_0 quant| <= 127, K=8 per
    fragment) so the accumulator is int32-EXACT.

    This is the vmadot INSTRUCTION as a first-class TYPED body op (I5), NOT the
    opaque hand-written asm helper: emission reads the op identity + the ime_op /
    MAC-fragment facts, never a name. The vmadot instruction's numeric semantics
    are proven bit-exact on real K1 (the M1b board seal); this op models the
    int8->int32 MAC contract the seal validates. For the bounded surface ime_op is
    limited to "vmadot", elem_in_bits to 8, accum_bits to 32.
  }];
  let arguments = (ins
    AnyType:$a_fragment,
    AnyType:$b_fragment,
    AnyType:$block_index,
    AnyType:$acc_in,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k
  );
  let results = (outs AnyType:$acc_out);
  let assemblyFormat = [{
    $a_fragment `,` $b_fragment `,` $block_index `,` $acc_in attr-dict `:`
    `(` type($a_fragment) `,` type($b_fragment) `,` type($block_index) `,`
    type($acc_in) `)` `->` type($acc_out)
  }];
  let hasVerifier = 1;
}

def Q40MatMulTileYieldOp : TCRVIME_Op<"q4_0_matmul_tile_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::ime::Q40MatMulTileOp">]> {
  let summary =
      "terminator naming the carried-out 4x4 int32 accumulator tile of the "
      "q4_0 IME GEMM tile region";
  let description = [{
    Terminates the tcrv.ime.q4_0_matmul_tile region by naming the carried-out 4x4
    int32 accumulator tile `acc_out` (the VmadotMacLeafOp result). It makes the
    loop-carried int32 dataflow explicit (the SSA value carried into the next
    contraction-block iteration), so the emitter maps it to the in-register int32
    accumulator with no implicit convention. It carries no compute, dtype, memory,
    or route semantics.
  }];
  let arguments = (ins Variadic<AnyType>:$acc_out);
  let assemblyFormat = "$acc_out attr-dict `:` type($acc_out)";
  let hasVerifier = 1;
}

def Q40MatMulTileOp : TCRVIME_Op<"q4_0_matmul_tile",
    [SingleBlock,
     DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary =
      "IME format-keyed q4_0 whole-matrix vmadot GEMM tile boundary (typed region)";
  let description = [{
    The selected IME execution boundary for a FORMAT-KEYED q4_0 whole-matrix
    int8->int32 GEMM: C[M,N] += dequant_q4_0(Bq4)[N,K] applied to A[M,K] int8
    activations, realized by tiling the FOUNDATION-validated 4x4x8 `vmadot`
    fragment over the (M/mac_m) x (N/mac_n) x (K/mac_k) grid, with the q4_0 weight
    matrix decoded (offset-binary nibble) into the int8 MAC fragment per
    contraction block. This is the format-keyed sibling of tcrv.ime.matmul (which
    is format-AGNOSTIC pre-packed int8): it carries the same capability-derived MAC
    envelope + problem dims (mat_m/mat_n/mat_k) PLUS the q4_0 weight-format FACTS
    (weight_format="q4_0", qk=32, the 18-byte weight_block_stride, the +2
    weight_quant_byte_offset).

    Unlike the flat boundary ops above, this op OWNS a single-block typed region
    (the M-FLAT typed-region front door, the RVV lowerToRepackGemm precedent
    applied to the IME matrix paradigm): the region is the INNERMOST
    contraction-block tile body, whose entry arguments are the `block_index`
    induction variable (index), the loaded 4x8 int8 activation fragment
    `a_fragment` (vector<32xi8>), and the carried-IN 4x4 int32 accumulator tile
    `acc_in` (vector<16xi32>). The region contains exactly the DECOMPOSED typed
    bricks -- one tcrv.ime.q4_0_dequant_core (the q4_0 weight decode), one
    tcrv.ime.vmadot_mac_leaf (the int8->int32 MAC), and the
    tcrv.ime.q4_0_matmul_tile_yield terminator (naming the carried-out int32 tile)
    -- and NO opaque hand helper. The emitter wraps the outer row/col/K loops
    around the region and binds the weight/activation/output pointers (op-identity
    driven, I5).

    A fail-closed verifier (I7) admits ONLY the validated IME1 int8->int32 vmadot
    envelope, the whole q4_0 problem dims (whole multiples of the MAC fragment; no
    remainder path), AND the exact three-brick region shape. It is an IME execution
    op, not a high-level matmul op: it carries the structural executable facts so
    emission is structural; it makes no correctness/performance claim in the IR
    (correctness is proven int32-exact + the vmadot bit-exact K1 seal). The
    coverage cell gemm_tile/q4_0@ime stays blocked-on-IME until the K1 seal.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    I64Attr:$mat_m,
    I64Attr:$mat_n,
    I64Attr:$mat_k,
    StrAttr:$weight_format,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_quant_byte_offset,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// G4 M2: the FORMAT-KEYED q8_0 IME GEMM tile (the SECOND format-keyed IME tile,
// the copy-adapt sibling of q4_0 above). q8_0 is FLAT int8: the ggml block is
// (fp16 d, int8 qs[32]) = 34 bytes, so the DECODE is a DIRECT int8 read (no
// nibble unpack, no offset-binary centering) -- strictly simpler than q4_0. The
// vmadot MAC leaf (VmadotMacLeafOp above) is REUSED verbatim (its validated
// envelope already admits |q8_0 quant| <= 127). Everything else -- the tiled
// int8->int32 MAC, the in-register int32 accumulate, the typed three-brick
// region -- mirrors q4_0. Correctness is int32-EXACT (the direct int8 quant is
// the COMPLETE integer content of q8_0; only the per-block fp16 scale `d` fold
// remains, a separate downstream epilogue exactly as q4_0). BOARD-PENDING until
// the K1 seal: gemm_tile/q8_0@ime stays blocked-on-IME until the silicon seal.
//===----------------------------------------------------------------------===//

def Q80DequantCoreOp : TCRVIME_Op<"q8_0_dequant_core"> {
  let summary = "IME q8_0 -> int8 DIRECT-read dequant core brick";
  let description = [{
    The typed q8_0 WEIGHT DECODE primitive of the format-keyed IME GEMM tile (the
    FLAT-int8 sibling of q4_0_dequant_core). It takes the per-tile
    contraction-block induction variable `block_index` and produces the decoded
    4x8 int8 B-fragment (a builtin vector<32xi8>): the ggml q8_0 block is
    (fp16 d, int8 qs[32]) = 34 bytes carrying 32 int8 quants; each int8 quant in
    [-128,127] is copied DIRECTLY into the decoded fragment (no nibble unpack, no
    offset-binary centering -- q8_0 is already flat int8), so the fragment feeds
    the int8->int32 vmadot MAC with NO precision loss. The per-block fp16 scale
    `d` is NOT applied here (a separate float fold AFTER the int32 MAC), so the
    decode is a PURE integer transform and the produced fragment is
    int32-MAC-exact.

    The op carries the bounded q8_0 layout FACTS (decode_model, qk=32, the 34-byte
    weight_block_stride, the +2 weight_quant_byte_offset past the fp16 d, and the
    +0 weight_scale_byte_offset) so emission is structural (I5): the emitter reads
    the op identity + these facts, never a q4/q8 name. For the bounded surface
    decode_model is limited to "q8_0_direct_int8".
  }];
  let arguments = (ins
    AnyType:$block_index,
    StrAttr:$decode_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$weight_scale_byte_offset
  );
  let results = (outs AnyType:$b_fragment);
  let assemblyFormat =
      "$block_index attr-dict `:` type($block_index) `->` type($b_fragment)";
  let hasVerifier = 1;
}

def Q80MatMulTileYieldOp : TCRVIME_Op<"q8_0_matmul_tile_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::ime::Q80MatMulTileOp">]> {
  let summary =
      "terminator naming the carried-out 4x4 int32 accumulator tile of the "
      "q8_0 IME GEMM tile region";
  let description = [{
    Terminates the tcrv.ime.q8_0_matmul_tile region by naming the carried-out 4x4
    int32 accumulator tile `acc_out` (the VmadotMacLeafOp result). It makes the
    loop-carried int32 dataflow explicit (the SSA value carried into the next
    contraction-block iteration), so the emitter maps it to the in-register int32
    accumulator with no implicit convention. It carries no compute, dtype, memory,
    or route semantics.
  }];
  let arguments = (ins Variadic<AnyType>:$acc_out);
  let assemblyFormat = "$acc_out attr-dict `:` type($acc_out)";
  let hasVerifier = 1;
}

def Q80MatMulTileOp : TCRVIME_Op<"q8_0_matmul_tile",
    [SingleBlock,
     DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary =
      "IME format-keyed q8_0 whole-matrix vmadot GEMM tile boundary (typed region)";
  let description = [{
    The selected IME execution boundary for a FORMAT-KEYED q8_0 whole-matrix
    int8->int32 GEMM: C[M,N] += dequant_q8_0(Bq8)[N,K] applied to A[M,K] int8
    activations, realized by tiling the FOUNDATION-validated 4x4x8 `vmadot`
    fragment over the (M/mac_m) x (N/mac_n) x (K/mac_k) grid, with the q8_0 weight
    matrix decoded (DIRECT int8 read) into the int8 MAC fragment per contraction
    block. This is the FLAT-int8 format-keyed sibling of tcrv.ime.q4_0_matmul_tile
    (which offset-binary-unpacks nibbles): it carries the same capability-derived
    MAC envelope + problem dims (mat_m/mat_n/mat_k) PLUS the q8_0 weight-format
    FACTS (weight_format="q8_0", qk=32, the 34-byte weight_block_stride, the +2
    weight_quant_byte_offset).

    Like q4_0_matmul_tile, this op OWNS a single-block typed region (the M-FLAT
    typed-region front door): the region is the INNERMOST contraction-block tile
    body, whose entry arguments are the `block_index` induction variable (index),
    the loaded 4x8 int8 activation fragment `a_fragment` (vector<32xi8>), and the
    carried-IN 4x4 int32 accumulator tile `acc_in` (vector<16xi32>). The region
    contains exactly the DECOMPOSED typed bricks -- one
    tcrv.ime.q8_0_dequant_core (the q8_0 weight decode), one
    tcrv.ime.vmadot_mac_leaf (the int8->int32 MAC, REUSED from q4_0), and the
    tcrv.ime.q8_0_matmul_tile_yield terminator (naming the carried-out int32 tile)
    -- and NO opaque hand helper. The emitter wraps the outer row/col/K loops
    around the region and binds the weight/activation/output pointers (op-identity
    driven, I5).

    A fail-closed verifier (I7) admits ONLY the validated IME1 int8->int32 vmadot
    envelope, the whole q8_0 problem dims (whole multiples of the MAC fragment; no
    remainder path), AND the exact three-brick region shape. It is an IME execution
    op, not a high-level matmul op: it carries the structural executable facts so
    emission is structural; it makes no correctness/performance claim in the IR
    (correctness is proven int32-exact + the vmadot bit-exact K1 seal). The
    coverage cell gemm_tile/q8_0@ime stays blocked-on-IME until the K1 seal.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,
    I64Attr:$elem_in_bits,
    I64Attr:$accum_bits,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    I64Attr:$mat_m,
    I64Attr:$mat_n,
    I64Attr:$mat_k,
    StrAttr:$weight_format,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_quant_byte_offset,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_IME_IR_IMEOPS_TD
