#ifndef WEFT_DIALECT_IME_IR_IMEOPS_TD
#define WEFT_DIALECT_IME_IR_IMEOPS_TD

include "mlir/IR/OpBase.td"
include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td"

def WEFT_IME_Dialect : Dialect {
  let name = "weft_ime";
  let cppNamespace = "::weft::ime";
  let summary = "Weft-RV IME (Integer Matrix Extension) extension dialect";
  let description = [{
    The IME dialect is the plugin-local extension-family surface for the
    Spacemit X60 Integer Matrix Extension (IME1). The architectural family is
    weft.ime; the concrete MLIR dialect namespace is weft_ime because MLIR
    dialect namespaces cannot contain '.' characters.

    IME is a non-RVV, RISC-V vendor matrix extension that rides the SAME unified
    WEFT common pipeline RVV uses (capability-driven proposal / legality /
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

class WEFTIME_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_IME_Dialect, mnemonic, traits>;

def MMAOp : WEFTIME_Op<"mma",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "IME int8->int32 matrix-multiply-accumulate (vmadot) boundary";
  let description = [{
    The selected IME execution boundary for a chosen weft.exec.variant: an
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

def MMAUOp : WEFTIME_Op<"mma_u",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "IME uint8->int32 matrix-multiply-accumulate (vmadotu) boundary";
  let description = [{
    The SECOND IME execution boundary (plugin breadth, N2): the UNSIGNED
    int8 form of the MAC unit, realized by the IME1 `vmadotu` instruction.
    Identical fragment-shape envelope to weft.ime.mma (capability-derived from
    VLEN/SEW: at VLEN=256/SEW=8 the X60 unit is 4x4x8, int32 accumulator), but a
    GENUINELY DIFFERENT emitted instruction: `vmadotu` (unsigned*unsigned, IME1
    signedness variant) computes C += A . B^T treating A and B as UNSIGNED
    int8, vs weft.ime.mma's signed `vmadot`. The two have distinct encodings
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

def MMASUOp : WEFTIME_Op<"mma_su",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "IME mixed-sign int8->int32 matrix-multiply-accumulate (vmadotsu) boundary";
  let description = [{
    The FOURTH IME execution boundary (plugin breadth, N2 rapid-add): the
    MIXED-SIGN form of the MAC unit, realized by the IME1 `vmadotsu`
    instruction. This is the canonical quantized mixed-sign case: A is treated
    as SIGNED int8 (activations) and B as UNSIGNED int8 (weights), C += A . B^T
    accumulated in int32. Identical fragment-shape envelope to weft.ime.mma
    (capability-derived from VLEN/SEW: at VLEN=256/SEW=8 the X60 unit is 4x4x8,
    int32 accumulator), but a GENUINELY DIFFERENT emitted instruction:
    `vmadotsu` (signed*unsigned int8 MAC, IME1 mixed-sign signedness variant)
    computes a numerically different result than weft.ime.mma's `vmadot`
    (signed*signed) and weft.ime.mma_u's `vmadotu` (unsigned*unsigned). The
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

def MMAUSOp : WEFTIME_Op<"mma_us",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "IME mixed-sign int8->int32 matrix-multiply-accumulate (vmadotus) boundary";
  let description = [{
    The SIXTH IME execution boundary (plugin breadth, N2 rapid-add): the OTHER
    MIXED-SIGN form of the MAC unit, realized by the IME1 `vmadotus`
    instruction — the reversed-order sibling of weft.ime.mma_su. Here A is
    treated as UNSIGNED int8 and B as SIGNED int8, C += A . B^T accumulated in
    int32. This completes the signedness family {ss=vmadot, uu=vmadotu,
    su=vmadotsu, us=vmadotus}. Identical fragment-shape envelope to weft.ime.mma
    (capability-derived from VLEN/SEW: at VLEN=256/SEW=8 the X60 unit is 4x4x8,
    int32 accumulator), but a GENUINELY DIFFERENT emitted instruction:
    `vmadotus` (unsigned A * signed B int8 MAC, IME1 mixed-sign signedness
    variant) computes a numerically different result than weft.ime.mma_su's
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

def MMASlideOp : WEFTIME_Op<"mma_slide",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
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

def MatMulOp : WEFTIME_Op<"matmul",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "IME int8->int32 tiled whole-matrix multiply-accumulate boundary";
  let description = [{
    The selected IME execution boundary for a WHOLE-MATRIX int8->int32
    multiply-accumulate C[M,N] += A[M,K] . B[K,N] realized by tiling the IME1
    MAC fragment (`vmadot` / `vmadotu`) over the (M/mac_m) x (N/mac_n) x
    (K/mac_k) tile grid. This is the same capability-derived MAC fragment
    (4x4x8 at VLEN=256/SEW=8) as weft.ime.mma, but applied to a problem-sized
    matmul rather than a single fragment — so it carries the problem dims
    (mat_m/mat_n/mat_k) IN ADDITION to the fragment shape (mac_m/mac_n/mac_k).

    A is stored (M,K) int8 row-major; B is stored (N,K) int8 row-major (== B^T
    of the math (K,N) matrix — the IME-native B^T layout); C is (M,N) int32
    row-major. The kernel keeps each 4x4 C output tile in the int32 VD pair
    across the WHOLE K reduction (the MAC accumulates in-register), storing C
    once per tile — so the per-output reduction the RVV path needs (vredsum) is
    structurally absent. The signedness form (`vmadot` signed / `vmadotu`
    unsigned) follows the same capability-derived fact as weft.ime.mma/mma_u.

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
// already int8 (the tiled weft.ime.matmul consumes pre-packed int8 A/B). The
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

def Q40DequantCoreOp : WEFTIME_Op<"q4_0_dequant_core"> {
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

def VmadotMacLeafOp : WEFTIME_Op<"vmadot_mac_leaf"> {
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

def Q40MatMulTileYieldOp : WEFTIME_Op<"q4_0_matmul_tile_yield",
    [Terminator,
     HasParent<"::weft::ime::Q40MatMulTileOp">]> {
  let summary =
      "terminator naming the carried-out 4x4 int32 accumulator tile of the "
      "q4_0 IME GEMM tile region";
  let description = [{
    Terminates the weft.ime.q4_0_matmul_tile region by naming the carried-out 4x4
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

def Q40MatMulTileOp : WEFTIME_Op<"q4_0_matmul_tile",
    [SingleBlock,
     DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary =
      "IME format-keyed q4_0 whole-matrix vmadot GEMM tile boundary (typed region)";
  let description = [{
    The selected IME execution boundary for a FORMAT-KEYED q4_0 whole-matrix
    int8->int32 GEMM: C[M,N] += dequant_q4_0(Bq4)[N,K] applied to A[M,K] int8
    activations, realized by tiling the FOUNDATION-validated 4x4x8 `vmadot`
    fragment over the (M/mac_m) x (N/mac_n) x (K/mac_k) grid, with the q4_0 weight
    matrix decoded (offset-binary nibble) into the int8 MAC fragment per
    contraction block. This is the format-keyed sibling of weft.ime.matmul (which
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
    bricks -- one weft.ime.q4_0_dequant_core (the q4_0 weight decode), one
    weft.ime.vmadot_mac_leaf (the int8->int32 MAC), and the
    weft.ime.q4_0_matmul_tile_yield terminator (naming the carried-out int32 tile)
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
    OptionalAttr<I64Attr>:$mac_batched,
    OptionalAttr<I64Attr>:$wide_njw,
    OptionalAttr<I64Attr>:$wide_vlen_bits,
    OptionalAttr<I64Attr>:$wide_input_fragment_vregs,
    OptionalAttr<I64Attr>:$wide_accumulator_vregs,
    OptionalAttr<I64Attr>:$wide_vreg_floor,
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

def Q80DequantCoreOp : WEFTIME_Op<"q8_0_dequant_core"> {
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

def Q80MatMulTileYieldOp : WEFTIME_Op<"q8_0_matmul_tile_yield",
    [Terminator,
     HasParent<"::weft::ime::Q80MatMulTileOp">]> {
  let summary =
      "terminator naming the carried-out 4x4 int32 accumulator tile of the "
      "q8_0 IME GEMM tile region";
  let description = [{
    Terminates the weft.ime.q8_0_matmul_tile region by naming the carried-out 4x4
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

def Q80MatMulTileOp : WEFTIME_Op<"q8_0_matmul_tile",
    [SingleBlock,
     DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary =
      "IME format-keyed q8_0 whole-matrix vmadot GEMM tile boundary (typed region)";
  let description = [{
    The selected IME execution boundary for a FORMAT-KEYED q8_0 whole-matrix
    int8->int32 GEMM: C[M,N] += dequant_q8_0(Bq8)[N,K] applied to A[M,K] int8
    activations, realized by tiling the FOUNDATION-validated 4x4x8 `vmadot`
    fragment over the (M/mac_m) x (N/mac_n) x (K/mac_k) grid, with the q8_0 weight
    matrix decoded (DIRECT int8 read) into the int8 MAC fragment per contraction
    block. This is the FLAT-int8 format-keyed sibling of weft.ime.q4_0_matmul_tile
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
    weft.ime.q8_0_dequant_core (the q8_0 weight decode), one
    weft.ime.vmadot_mac_leaf (the int8->int32 MAC, REUSED from q4_0), and the
    weft.ime.q8_0_matmul_tile_yield terminator (naming the carried-out int32 tile)
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
    OptionalAttr<I64Attr>:$mac_batched,
    OptionalAttr<I64Attr>:$wide_njw,
    OptionalAttr<I64Attr>:$wide_vlen_bits,
    OptionalAttr<I64Attr>:$wide_input_fragment_vregs,
    OptionalAttr<I64Attr>:$wide_accumulator_vregs,
    OptionalAttr<I64Attr>:$wide_vreg_floor,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// G4 M2b: the FORMAT-KEYED q4_K IME GEMM tile (the SUPER-BLOCK K-quant tile, the
// DEDICATED effort beyond the q4_0/q8_0 single-decode copy-adapt). q4_K is a
// super-block format: the ggml block is 256 weights / 144 bytes = (fp16 d, fp16
// dmin, 12-byte 6-bit packed scales/mins for 8 sub-blocks, 128-byte 4-bit
// quants). The bare vmadot int8 MAC computes ONLY the raw-nibble contraction
// Sum(A_i*q_i); sealing ONLY that is a HOLLOW q4_K representation (it drops the
// per-sub-block scale weighting + the min bias that DEFINE q4_K). The faithful
// int32-exact core (the board-seal object) is the TWO-LEVEL fold, mirroring the
// RVV kquant_dmin_bsums_min precedent:
//
//   S_scale = Sum_b sc_b * (Sum_{i in b} A_i * q_i)     (int32-exact)
//   S_min   = Sum_b m_b  * (Sum_{i in b} A_i)           (int32-exact)
//   C       = d * S_scale - dmin * S_min                (deferred fp16 float epilogue)
//
// where sc_b / m_b are the 6-bit per-sub-block scale / min (get_scale_min_k4),
// sumi_b = the per-sub-block vmadot MAC result, asum_b = the per-sub-block
// activation sum. This tile therefore carries THREE NEW typed bricks beyond the
// reused vmadot MAC leaf: the 6-bit scale/min bit-unpack, the per-sub-block
// scale-weighted accumulate (S_scale), and the activation-sum min-bias
// accumulate (S_min); plus a q4_K RAW-nibble decode core (q4_K nibbles are
// UNSIGNED [0,15], NOT the q4_0 offset-binary [-8,7]). The d/dmin fp16 fold is
// the SOLE deferred float epilogue (exactly the q4_0/q8_0 scalar-d epilogue,
// two-scalar here). BOARD-PENDING until the K1 seal: gemm_tile/q4_K@ime stays
// blocked-on-IME until the silicon seal validates the int32 core bit-exact.
//===----------------------------------------------------------------------===//

def Q4KDequantCoreOp : WEFTIME_Op<"q4_K_dequant_core"> {
  let summary = "IME q4_K -> int8 RAW-nibble (unsigned) dequant core brick";
  let description = [{
    The typed q4_K WEIGHT DECODE primitive of the format-keyed IME GEMM tile (the
    super-block sibling of q4_0_dequant_core). It takes the per-tile
    contraction-block induction variable `block_index` and produces the decoded
    4x8 int8 B-fragment (a builtin vector<32xi8>): a ggml q4_K super-block is 144
    bytes ([fp16 d][fp16 dmin][12-byte packed 6-bit scales/mins][128-byte 4-bit
    quants]) carrying 256 nibbles across 8 sub-blocks of 32. Each 4-bit nibble q in
    [0,15] decodes to the RAW UNSIGNED int8 quant value q (NO offset-binary
    centering -- UNLIKE q4_0's q-8: the q4_K quant is x = d*sc*q - dmin*m, so the
    quant fed to the MAC is the raw non-negative nibble and the min bias is applied
    separately downstream), which fits int8 exactly, so the decoded fragment feeds
    the int8->int32 vmadot MAC with NO precision loss. Neither the per-sub-block
    6-bit scale `sc` nor the per-block fp16 scale `d` is applied here (both are
    separate downstream folds), so the decode is a PURE integer transform and the
    produced fragment is int32-MAC-exact.

    The op carries the bounded q4_K layout FACTS (decode_model, qk=256 weights per
    super-block, the 144-byte weight_block_stride, the +16 weight_quant_byte_offset
    past the fp16 d/dmin + 12-byte scales, and the +4 weight_scale_byte_offset) so
    emission is structural (I5). For the bounded surface decode_model is limited to
    "q4_K_raw_nibble".
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

def Q4KScaleMinUnpackCoreOp : WEFTIME_Op<"q4_K_scale_min_unpack_core"> {
  let summary = "IME q4_K 6-bit per-sub-block scale/min bit-unpack brick (get_scale_min_k4)";
  let description = [{
    The typed q4_K SCALE/MIN UNPACK primitive (the FIRST of the three NEW q4_K
    bricks). It takes the per-tile contraction-block induction variable
    `block_index` and produces the two per-sub-block 6-bit quantities as int32 tile
    vectors: `sc` (the 8 per-sub-block scales sc_b) and `m` (the 8 per-sub-block
    mins m_b), unpacked from the 12-byte packed scales region by the canonical ggml
    `get_scale_min_k4` bit-dance (sub-blocks 0..3: low-6-bit of scales[j] /
    scales[j+4]; sub-blocks 4..7: a 4-bit-plus-2-high-bit reassembly). Both sc_b and
    m_b are 6-bit unsigned in [0,63]. This is a PURE integer transform of the packed
    bytes; the sc weighting and the m bias are applied by the two accumulate bricks
    below.

    The op carries the bounded q4_K scale-layout FACTS (scale_min_model, the 8
    num_sub_blocks, the 6 scale_bits, the 12-byte k_scale_size, and the +4
    weight_scale_byte_offset). For the bounded surface scale_min_model is limited to
    "get_scale_min_k4".
  }];
  let arguments = (ins
    AnyType:$block_index,
    StrAttr:$scale_min_model,
    I64Attr:$num_sub_blocks,
    I64Attr:$scale_bits,
    I64Attr:$k_scale_size,
    I64Attr:$weight_scale_byte_offset
  );
  let results = (outs AnyType:$sc, AnyType:$m);
  let assemblyFormat =
      "$block_index attr-dict `:` type($block_index) `->` type($sc) `,` type($m)";
  let hasVerifier = 1;
}

def Q4KScaleWeightedAccumOp : WEFTIME_Op<"q4_K_scale_weighted_accum"> {
  let summary = "IME q4_K per-sub-block scale-weighted accumulate brick (S_scale += sc_b*sumi_b)";
  let description = [{
    The typed q4_K SCALE-WEIGHTED ACCUMULATE primitive (the SECOND of the three NEW
    q4_K bricks -- the part a bare-nibble-MAC hollow cert would DROP). It takes the
    per-sub-block vmadot MAC result `sumi_b` (vector<16xi32>, the raw int32
    contraction Sum_{i in b} A_i*q_i produced by the reused vmadot_mac_leaf), the
    per-sub-block 6-bit scale `sc_b` (vector<16xi32>, from the scale/min unpack
    brick), and the carried-IN int32 scale accumulator tile `acc_scale_in`
    (vector<16xi32>), and produces the carried-OUT scale accumulator
    `acc_scale_out` = acc_scale_in + sc_b * sumi_b (int32-EXACT). Over the 8
    sub-blocks this realizes S_scale = Sum_b sc_b * sumi_b. There is NO overflow in
    the validated envelope (|sc_b| <= 63, |sumi_b| <= 32*128*15, 8 sub-blocks;
    S_scale <= ~3.1e7 << INT32_MAX), so the accumulator is int32-EXACT.

    For the bounded surface accum_model is limited to "scale_weighted_sum".
  }];
  let arguments = (ins
    AnyType:$sumi,
    AnyType:$sc,
    AnyType:$acc_scale_in,
    StrAttr:$accum_model
  );
  let results = (outs AnyType:$acc_scale_out);
  let assemblyFormat = [{
    $sumi `,` $sc `,` $acc_scale_in attr-dict `:`
    `(` type($sumi) `,` type($sc) `,` type($acc_scale_in) `)` `->` type($acc_scale_out)
  }];
  let hasVerifier = 1;
}

def Q4KMinBiasAccumOp : WEFTIME_Op<"q4_K_min_bias_accum"> {
  let summary = "IME q4_K activation-sum min-bias accumulate brick (S_min += m_b*asum_b)";
  let description = [{
    The typed q4_K MIN-BIAS ACCUMULATE primitive (the THIRD of the three NEW q4_K
    bricks -- the OTHER part a bare-nibble-MAC hollow cert would DROP). The q4_K
    min bias needs the PURE activation sub-block sum asum_b = Sum_{i in b} A_i,
    which the vmadot MAC leaf CANNOT express (vmadot always contracts A against a
    weight fragment), so it is an INDEPENDENT reduce. This op takes the int8
    activation fragment `a_fragment` (vector<32xi8>, whose per-row lanes it reduces
    to asum_b), the per-sub-block 6-bit min `m_b` (vector<16xi32>, from the
    scale/min unpack brick), and the carried-IN int32 min accumulator tile
    `acc_min_in` (vector<16xi32>), and produces `acc_min_out` = acc_min_in + m_b *
    asum_b (int32-EXACT). Over the 8 sub-blocks this realizes S_min = Sum_b m_b *
    asum_b, which the float epilogue subtracts as dmin * S_min. There is NO overflow
    in the validated envelope (|m_b| <= 63, |asum_b| <= 32*128, 8 sub-blocks; S_min
    <= ~2.1e6 << INT32_MAX).

    For the bounded surface bias_model is limited to "activation_sum_min_bias".
  }];
  let arguments = (ins
    AnyType:$a_fragment,
    AnyType:$m,
    AnyType:$acc_min_in,
    StrAttr:$bias_model
  );
  let results = (outs AnyType:$acc_min_out);
  let assemblyFormat = [{
    $a_fragment `,` $m `,` $acc_min_in attr-dict `:`
    `(` type($a_fragment) `,` type($m) `,` type($acc_min_in) `)` `->` type($acc_min_out)
  }];
  let hasVerifier = 1;
}

def Q4KMatMulTileYieldOp : WEFTIME_Op<"q4_K_matmul_tile_yield",
    [Terminator,
     HasParent<"::weft::ime::Q4KMatMulTileOp">]> {
  let summary =
      "terminator naming the carried-out (S_scale, S_min) int32 accumulator tiles "
      "of the q4_K IME GEMM tile region";
  let description = [{
    Terminates the weft.ime.q4_K_matmul_tile region by naming the TWO carried-out
    4x4 int32 accumulator tiles: `acc_out` = (acc_scale_out, acc_min_out) -- the
    S_scale scale-weighted accumulator and the S_min min-bias accumulator carried
    into the next contraction-block iteration. It makes the loop-carried TWO-LEVEL
    int32 dataflow explicit (both the scale accumulator and the min accumulator), so
    the emitter maps them to the two in-register int32 accumulators with no implicit
    convention. It carries no compute, dtype, memory, or route semantics.
  }];
  let arguments = (ins Variadic<AnyType>:$acc_out);
  let assemblyFormat = "$acc_out attr-dict `:` type($acc_out)";
  let hasVerifier = 1;
}

def Q4KMatMulTileOp : WEFTIME_Op<"q4_K_matmul_tile",
    [SingleBlock,
     DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary =
      "IME format-keyed q4_K super-block vmadot GEMM tile boundary (typed region)";
  let description = [{
    The selected IME execution boundary for a FORMAT-KEYED q4_K whole-matrix
    int8->int32 GEMM: C[M,N] += dequant_q4_K(Bq4K)[N,K] applied to A[M,K] int8
    activations, realized by tiling the FOUNDATION-validated 4x4x8 `vmadot` fragment
    over the (M/mac_m) x (N/mac_n) x (K/mac_k) grid, with the q4_K super-block
    weight matrix decoded (RAW unsigned nibble) into the int8 MAC fragment and the
    K-quant TWO-LEVEL 6-bit scale/min fold applied per super-block. This is the
    SUPER-BLOCK format-keyed sibling of weft.ime.q4_0_matmul_tile / q8_0_matmul_tile
    (the single-decode flat tiles): it carries the same capability-derived MAC
    envelope + problem dims (mat_m/mat_n/mat_k) PLUS the q4_K weight-format FACTS
    (weight_format="q4_K", qk=256, the 144-byte weight_block_stride, the +16
    weight_quant_byte_offset).

    Like q4_0_matmul_tile, this op OWNS a single-block typed region, BUT its region
    is the DEDICATED q4_K super-block body (NOT a copy-adapt of the flat tile): the
    entry args are the `block_index` induction variable (index), the loaded 4x8 int8
    activation fragment `a_fragment` (vector<32xi8>), and the carried-IN TWO int32
    accumulator tiles `acc_scale_in` + `acc_min_in` (vector<16xi32>). The region
    contains exactly the DECOMPOSED typed bricks -- one weft.ime.q4_K_dequant_core
    (the raw-nibble weight decode), one weft.ime.q4_K_scale_min_unpack_core (the
    6-bit scale/min bit-unpack), one weft.ime.vmadot_mac_leaf (the int8->int32 MAC,
    REUSED from q4_0), one weft.ime.q4_K_scale_weighted_accum (S_scale += sc_b*sumi_b),
    one weft.ime.q4_K_min_bias_accum (S_min += m_b*asum_b), and the
    weft.ime.q4_K_matmul_tile_yield terminator (naming the carried-out S_scale +
    S_min tiles) -- and NO opaque hand helper. The emitter wraps the outer
    row/col/super-block/sub-block loops around the region and binds the
    weight/activation/output pointers (op-identity driven, I5).

    A fail-closed verifier (I7) admits ONLY the validated IME1 int8->int32 vmadot
    envelope, the whole q4_K problem dims (whole multiples of the MAC fragment AND
    of the 256-weight super-block; no remainder path), AND the exact SIX-brick region
    shape (the raw-nibble decode + the scale/min unpack + the vmadot MAC + the
    scale-weighted accum + the min-bias accum + the two-tile yield). Dropping the
    scale-weighted-accum or the min-bias-accum brick (the hollow bare-MAC shape) is
    REJECTED. It is an IME execution op, not a high-level matmul op: it carries the
    structural executable facts so emission is structural; it makes no
    correctness/performance claim in the IR (correctness is proven int32-exact for
    the two-level S_scale/S_min core + the vmadot bit-exact K1 seal; the d/dmin fp16
    fold is a deferred deterministic float epilogue). The coverage cell
    gemm_tile/q4_K@ime stays blocked-on-IME until the K1 seal.
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
    OptionalAttr<I64Attr>:$mac_batched,
    OptionalAttr<I64Attr>:$wide_njw,
    OptionalAttr<I64Attr>:$wide_vlen_bits,
    OptionalAttr<I64Attr>:$wide_input_fragment_vregs,
    OptionalAttr<I64Attr>:$wide_accumulator_vregs,
    OptionalAttr<I64Attr>:$wide_vreg_floor,
    StrAttr:$available_harts,
    OptionalAttr<StrAttr>:$ime_reason
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_IME_IR_IMEOPS_TD
