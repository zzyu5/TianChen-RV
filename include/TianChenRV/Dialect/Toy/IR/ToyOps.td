#ifndef TIANCHENRV_DIALECT_TOY_IR_TOYOPS_TD
#define TIANCHENRV_DIALECT_TOY_IR_TOYOPS_TD

include "mlir/IR/OpBase.td"

def TCRV_Toy_Dialect : Dialect {
  let name = "tcrv_toy";
  let cppNamespace = "::tianchenrv::tcrv::toy";
  let summary = "TianChen-RV Toy extension metadata dialect";
  let description = [{
    The Toy dialect is a plugin-local extension template surface. The
    architectural family is tcrv.toy; the concrete MLIR dialect namespace is
    tcrv_toy because MLIR dialect namespaces cannot contain '.' characters.

    This dialect intentionally exposes only selected lowering-boundary metadata
    for plugin integration tests. It does not model tensor/tile computation,
    executable lowering, runtime calls, binary artifacts, correctness, or
    performance.
  }];
}

class TCRVToy_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Toy_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVToy_Op<"lowering_boundary", []> {
  let summary = "metadata-only Toy selected lowering boundary";
  let description = [{
    Records the Toy plugin-owned selected lowering boundary for a chosen
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, metadata-only status, required capability
    references, and Toy template metadata.

    This op is not a compute operation and is not an executable lowering step.
    It exists to prove that a future extension plugin can own a dialect-local
    boundary while generic core passes route through the registry.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$template_abi,
    StrAttr:$handoff_kind,
    OptionalAttr<StrAttr>:$template_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_TOY_IR_TOYOPS_TD
