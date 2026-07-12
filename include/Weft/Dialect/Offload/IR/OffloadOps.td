#ifndef WEFT_DIALECT_OFFLOAD_IR_OFFLOADOPS_TD
#define WEFT_DIALECT_OFFLOAD_IR_OFFLOADOPS_TD

include "mlir/IR/OpBase.td"

def WEFT_Offload_Dialect : Dialect {
  let name = "weft_offload";
  let cppNamespace = "::weft::offload";
  let summary = "Weft-RV runtime-offload extension metadata dialect";
  let description = [{
    The offload dialect is the plugin-local extension surface for runtime
    offload handoff metadata. The architectural family is weft.offload; the
    concrete MLIR dialect namespace is weft_offload because MLIR dialect
    namespaces cannot contain '.' characters.

    This first slice intentionally exposes only selected runtime-offload
    boundary metadata. It does not model high-level tensor/tile computation,
    Sophgo-specific behavior, DMA/runtime library calls, accelerator kernels,
    executable lowering, object generation, correctness, or performance.
  }];
}

class WEFTOffload_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Offload_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : WEFTOffload_Op<"lowering_boundary", []> {
  let summary = "fail-closed runtime-offload lowering boundary";
  let description = [{
    Records the offload plugin-owned runtime handoff boundary for a selected
    weft.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, no-active-route status, required capability
    references, runtime ABI handoff metadata, and a bounded handoff reason.

    This op is not an accelerator compute operation. It does not lower to
    vendor runtime calls, allocate or copy buffers, generate objects, link
    libraries, execute hardware, prove correctness, or measure performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$runtime_abi,
    StrAttr:$handoff_kind,
    OptionalAttr<StrAttr>:$handoff_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_OFFLOAD_IR_OFFLOADOPS_TD
