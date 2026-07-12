#ifndef WEFT_DIALECT_TOY_IR_TOYOPS_TD
#define WEFT_DIALECT_TOY_IR_TOYOPS_TD

include "mlir/IR/OpBase.td"
include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td"

def WEFT_Toy_Dialect : Dialect {
  let name = "weft_toy";
  let cppNamespace = "::weft::toy";
  let summary = "Weft-RV Toy extension construction-template dialect";
  let description = [{
    The Toy dialect is a plugin-local extension template surface. The
    architectural family is weft.toy; the concrete MLIR dialect namespace is
    weft_toy because MLIR dialect namespaces cannot contain '.' characters.

    This dialect intentionally exposes a minimal ODS-backed role-op boundary
    for plugin integration tests. The compute skeleton can be consumed by the
    Toy plugin's bounded EmitC lowerable route, but it does not model
    tensor/tile computation, target object export, runtime correctness, or
    performance.
  }];
}

class WEFTToy_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Toy_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : WEFTToy_Op<"lowering_boundary", []> {
  let summary = "fail-closed Toy selected lowering boundary";
  let description = [{
    Records the Toy plugin-owned selected lowering boundary for a chosen
    weft.exec.variant. The op carries source kernel, selected variant, origin
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

def ComputeSkeletonOp : WEFTToy_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "minimal Toy ODS compute role-op boundary";
  let description = [{
    Records the minimal Toy extension-family compute role op in ODS. The op
    proves that a concrete non-RVV extension family can materialize a typed
    MLIR role operation implementing the generated EmitC-lowerable provenance
    interface and have Toy construction validation consume that IR identity.

    This op is a construction-protocol role boundary consumed by the bounded
    Toy EmitC route. It does not model tensor/tile computation, runtime ABI
    values, generated objects, correctness, or performance. Configure, load,
    and store are manifest-ordered staged roles for this minimal Toy slice.
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

#endif // WEFT_DIALECT_TOY_IR_TOYOPS_TD
