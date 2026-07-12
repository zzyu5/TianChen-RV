#ifndef WEFT_DIALECT_DEMO_IR_DEMOOPS_TD
#define WEFT_DIALECT_DEMO_IR_DEMOOPS_TD

include "mlir/IR/OpBase.td"
include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td"

def WEFT_Demo_Dialect : Dialect {
  let name = "weft_demo";
  let cppNamespace = "::weft::demo_ext";
  let summary = "Weft-RV Demo extension dialect";
  let description = [{
    The Demo dialect is a plugin-local extension demo surface. The
    architectural family is weft.demo; the concrete MLIR dialect namespace is
    weft_demo because MLIR dialect namespaces cannot contain '.' characters.

    This dialect exposes selected lowering-boundary metadata plus a minimal
    ODS-backed role-op boundary used by construction protocol tests. The
    compute_skeleton role can be consumed by the Demo plugin's bounded
    EmitC construction-demo route for generated C++ evidence, but it does
    not model tensor/tile computation, target runtime execution, hardware
    correctness, or performance.
  }];
}

class WEFTDemo_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Demo_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : WEFTDemo_Op<"lowering_boundary", []> {
  let summary = "fail-closed Demo selected lowering boundary";
  let description = [{
    Records the Demo plugin-owned selected lowering boundary for a chosen
    weft.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, no-active-route status, required capability
    references, and Demo extension metadata.

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
    OptionalAttr<StrAttr>:$demo_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def ComputeSkeletonOp : WEFTDemo_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "minimal Demo ODS compute role-op boundary";
  let description = [{
    Records the first real Demo extension-family role op in ODS. The op
    proves that a future extension can materialize a concrete MLIR role
    operation implementing the generated EmitC-lowerable provenance interface
    and have Demo construction validation consume that interface-backed IR
    identity.

    This op is a construction-demo compute role boundary only. It can be
    consumed by the Demo-owned EmitC route to prove the construction
    demo reaches generated C++ evidence, but it does not model tensor/tile
    computation, target runtime execution, generated objects, correctness, or
    performance. Configure, load, and store remain manifest-ordered staged
    roles for this slice.
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
    OptionalAttr<StrAttr>:$demo_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_DEMO_IR_DEMOOPS_TD
