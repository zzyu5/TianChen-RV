#ifndef TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD
#define TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD

include "mlir/IR/OpBase.td"

def TCRV_Template_Dialect : Dialect {
  let name = "tcrv_template";
  let cppNamespace = "::tianchenrv::tcrv::template_ext";
  let summary = "TianChen-RV Template extension metadata dialect";
  let description = [{
    The Template dialect is a plugin-local extension template surface. The
    architectural family is tcrv.template; the concrete MLIR dialect namespace is
    tcrv_template because MLIR dialect namespaces cannot contain '.' characters.

    This dialect intentionally exposes only selected lowering-boundary metadata
    for plugin integration tests. It does not model tensor/tile computation,
    executable lowering, runtime calls, binary artifacts, correctness, or
    performance.
  }];
}

class TCRVTemplate_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Template_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVTemplate_Op<"lowering_boundary", []> {
  let summary = "metadata-only Template selected lowering boundary";
  let description = [{
    Records the Template plugin-owned selected lowering boundary for a chosen
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, metadata-only status, required capability
    references, and Template extension metadata.

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
    StrAttr:$integration_contract,
    StrAttr:$handoff_kind,
    OptionalAttr<StrAttr>:$template_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD
