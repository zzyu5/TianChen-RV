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

#endif // TIANCHENRV_DIALECT_IME_IR_IMEOPS_TD
