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
    (vmadot=0xe210312b vs vmadotu=0xe210022b) and distinct numeric semantics; on
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
    three have distinct encodings (vmadot=0xe210312b, vmadotu=0xe210022b,
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

#endif // TIANCHENRV_DIALECT_IME_IR_IMEOPS_TD
