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

    This dialect exposes the family-local typed body constructed for a selected
    variant. The compute_skeleton is the construction result consumed by the
    Demo backend; it is not a metadata-only lowering boundary or a replay of a
    generic role manifest.
  }];
}

class WEFTDemo_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Demo_Dialect, mnemonic, traits>;

def ComputeSkeletonOp : WEFTDemo_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "Demo family-local typed computation body";
  let description = [{
    The Demo formula constructs this typed operation for one selected variant.
    Its MLIR type is the compute identity; string role, status, interface and
    capability mirrors are intentionally absent. The backend consumes this
    exact operation through WEFTEmitCLowerableOpInterface.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    OptionalAttr<StrAttr>:$demo_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_DEMO_IR_DEMOOPS_TD
