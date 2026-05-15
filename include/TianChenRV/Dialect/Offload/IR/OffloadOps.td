#ifndef TIANCHENRV_DIALECT_OFFLOAD_IR_OFFLOADOPS_TD
#define TIANCHENRV_DIALECT_OFFLOAD_IR_OFFLOADOPS_TD

include "mlir/IR/OpBase.td"

def TCRV_Offload_Dialect : Dialect {
  let name = "tcrv_offload";
  let cppNamespace = "::tianchenrv::tcrv::offload";
  let summary = "TianChen-RV runtime-offload extension metadata dialect";
  let description = [{
    The offload dialect is the plugin-local extension surface for runtime
    offload handoff metadata. The architectural family is tcrv.offload; the
    concrete MLIR dialect namespace is tcrv_offload because MLIR dialect
    namespaces cannot contain '.' characters.

    This first slice intentionally exposes only selected runtime-offload
    boundary metadata. It does not model high-level tensor/tile computation,
    Sophgo-specific behavior, DMA/runtime library calls, accelerator kernels,
    executable lowering, object generation, correctness, or performance.
  }];
}

class TCRVOffload_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Offload_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVOffload_Op<"lowering_boundary", []> {
  let summary = "fail-closed runtime-offload lowering boundary";
  let description = [{
    Records the offload plugin-owned runtime handoff boundary for a selected
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
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

#endif // TIANCHENRV_DIALECT_OFFLOAD_IR_OFFLOADOPS_TD
