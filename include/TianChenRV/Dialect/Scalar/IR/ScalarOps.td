#ifndef TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD
#define TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD

include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"

def TCRV_Scalar_Dialect : Dialect {
  let name = "tcrv_scalar";
  let cppNamespace = "::tianchenrv::tcrv::scalar";
  let summary = "TianChen-RV scalar fallback extension dialect";
  let description = [{
    The scalar fallback dialect namespace is the portable, no-RISC-V-extension
    fallback extension family. The architectural family is tcrv.scalar; the
    concrete MLIR dialect namespace is tcrv_scalar because MLIR dialect
    namespaces cannot contain '.' characters.

    This slice exposes one minimal ODS-backed compute role-op boundary
    (tcrv_scalar.compute_skeleton) that the scalar backend emission driver
    lowers directly to a standalone, portable-scalar EmitC module (no
    __riscv_ intrinsics). It is a TRIVIAL tracer-bullet compute boundary: it
    proves the scalar family can flow a typed role op through the shared
    typed-emission harness and the emitc->C++ translate route. It does not
    model tensor/tile computation, target object export, runtime correctness,
    or performance.
  }];
}

class TCRVScalar_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Scalar_Dialect, mnemonic, traits>;

def ComputeSkeletonOp : TCRVScalar_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface>]> {
  let summary = "minimal portable-scalar ODS compute role-op boundary";
  let description = [{
    Records the minimal scalar extension-family compute role op in ODS. The op
    proves that the portable scalar fallback family can materialize a typed
    MLIR role operation implementing the generated EmitC-lowerable provenance
    interface and have the scalar backend emission driver consume that IR
    identity to emit a pure-scalar (no __riscv_) EmitC module.

    The exported function name is derived from `source_kernel` and
    `selected_variant`, and the emitted body echoes `scalar_immediate`, so the
    emission is operand-driven (changing an attribute changes the emitted C).
    This is a TRIVIAL tracer-bullet boundary: it does not model tensor/tile
    computation, runtime ABI values, generated objects, correctness, or
    performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$scalar_immediate
  );

  let assemblyFormat = "attr-dict";
}

#endif // TIANCHENRV_DIALECT_SCALAR_IR_SCALAROPS_TD
