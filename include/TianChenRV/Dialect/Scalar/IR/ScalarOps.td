#ifndef TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD
#define TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD

include "mlir/IR/OpBase.td"

def TCRV_Scalar_Dialect : Dialect {
  let name = "tcrv_scalar";
  let cppNamespace = "::tianchenrv::tcrv::scalar";
  let summary = "TianChen-RV scalar fallback extension dialect";
  let description = [{
    The scalar fallback dialect namespace is reserved for a future
    extension-family scalar fallback rebuild. The architectural family is
    tcrv.scalar; the concrete MLIR dialect namespace is tcrv_scalar because
    MLIR dialect namespaces cannot contain '.' characters.

    This deletion-campaign slice intentionally exposes no active operations.
    It does not model scalar arithmetic, selected-boundary metadata,
    source-export bodies, LLVM lowering, runtime ABI glue, generated
    artifacts, correctness, or performance. Historical scalar microkernel
    source-export ops and metadata-only lowering-boundary ops are deleted
    legacy route residue and are not part of the active dialect surface.
  }];
}

class TCRVScalar_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Scalar_Dialect, mnemonic, traits>;

#endif // TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD
