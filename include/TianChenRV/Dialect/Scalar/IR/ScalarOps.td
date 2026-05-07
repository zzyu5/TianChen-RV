#ifndef TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD
#define TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD

include "mlir/IR/OpBase.td"

def TCRV_Scalar_Dialect : Dialect {
  let name = "tcrv_scalar";
  let cppNamespace = "::tianchenrv::tcrv::scalar";
  let summary = "TianChen-RV scalar fallback extension metadata dialect";
  let description = [{
    The scalar fallback dialect is a plugin-local extension surface for
    selected portable fallback metadata. The architectural family is
    tcrv.scalar; the concrete MLIR dialect namespace is tcrv_scalar because
    MLIR dialect namespaces cannot contain '.' characters.

    This first slice intentionally exposes only non-compute selected-boundary
    metadata. It does not model scalar arithmetic, memory loops, LLVM lowering,
    runtime ABI glue, generated artifacts, correctness, or performance.
  }];
}

class TCRVScalar_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Scalar_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVScalar_Op<"lowering_boundary", []> {
  let summary = "metadata-only scalar fallback lowering boundary";
  let description = [{
    Records the scalar fallback plugin-owned lowering boundary for a selected
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, metadata-only status, required capability
    references, and an optional fallback reason so later scalar lowering work
    has a stable plugin-local attachment point.

    This op is not a scalar compute operation. It does not lower to LLVM, emit
    scalar loops, create a runtime ABI, generate artifacts, execute kernels,
    prove correctness, or measure performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    OptionalAttr<StrAttr>:$fallback_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD
