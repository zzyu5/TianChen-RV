#ifndef TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD
#define TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD

include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"

def TCRV_Template_Dialect : Dialect {
  let name = "tcrv_template";
  let cppNamespace = "::tianchenrv::tcrv::template_ext";
  let summary = "TianChen-RV Template extension dialect";
  let description = [{
    The Template dialect is a plugin-local extension template surface. The
    architectural family is tcrv.template; the concrete MLIR dialect namespace is
    tcrv_template because MLIR dialect namespaces cannot contain '.' characters.

    This dialect exposes selected lowering-boundary metadata plus a minimal
    ODS-backed role-op boundary used by construction protocol tests. It does
    not model tensor/tile computation, executable lowering, runtime calls,
    binary artifacts, correctness, or performance.
  }];
}

class TCRVTemplate_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Template_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVTemplate_Op<"lowering_boundary", []> {
  let summary = "fail-closed Template selected lowering boundary";
  let description = [{
    Records the Template plugin-owned selected lowering boundary for a chosen
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, no-active-route status, required capability
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

def ComputeSkeletonOp : TCRVTemplate_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "minimal Template ODS compute role-op boundary";
  let description = [{
    Records the first real Template extension-family role op in ODS. The op
    proves that a future extension can materialize a concrete MLIR role
    operation implementing the generated EmitC-lowerable provenance interface
    and have Template construction validation consume that interface-backed IR
    identity.

    This op is a construction-template compute role boundary only. It does not
    model tensor/tile computation, executable lowering, runtime ABI values,
    generated objects, correctness, or performance. Configure, load, and store
    remain manifest-ordered staged roles for this slice.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$typed_role,
    I64Attr:$role_order,
    StrAttr:$source_role,
    StrAttr:$role_specific_interface,
    OptionalAttr<StrAttr>:$template_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD
