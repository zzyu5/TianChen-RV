#ifndef TIANCHENRV_DIALECT_TOY_IR_TOYOPS_TD
#define TIANCHENRV_DIALECT_TOY_IR_TOYOPS_TD

include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"

def TCRV_Toy_Dialect : Dialect {
  let name = "tcrv_toy";
  let cppNamespace = "::tianchenrv::tcrv::toy";
  let summary = "TianChen-RV Toy extension metadata dialect";
  let description = [{
    The Toy dialect is a plugin-local extension template surface. The
    architectural family is tcrv.toy; the concrete MLIR dialect namespace is
    tcrv_toy because MLIR dialect namespaces cannot contain '.' characters.

    This dialect intentionally exposes only selected lowering-boundary metadata
    plus a minimal ODS-backed role-op boundary for plugin integration tests. It
    does not model tensor/tile computation, executable lowering, runtime calls,
    binary artifacts, correctness, or performance.
  }];
}

class TCRVToy_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Toy_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVToy_Op<"lowering_boundary", []> {
  let summary = "fail-closed Toy selected lowering boundary";
  let description = [{
    Records the Toy plugin-owned selected lowering boundary for a chosen
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, no-active-route status, required capability
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

def ComputeSkeletonOp : TCRVToy_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "minimal Toy ODS compute role-op boundary";
  let description = [{
    Records the minimal Toy extension-family compute role op in ODS. The op
    proves that a concrete non-RVV extension family can materialize a typed
    MLIR role operation implementing the generated EmitC-lowerable provenance
    interface and have Toy construction validation consume that IR identity.

    This op is a construction-protocol role boundary only. It does not model
    tensor/tile computation, executable lowering, runtime ABI values, generated
    objects, correctness, or performance. Configure, load, and store are
    manifest-ordered staged roles for this minimal Toy slice.
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

#endif // TIANCHENRV_DIALECT_TOY_IR_TOYOPS_TD
